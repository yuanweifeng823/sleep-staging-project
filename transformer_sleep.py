import warnings

warnings.filterwarnings('ignore')
import mne
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 全局参数
DATA_PATH = "./data/"
WIN_SIZE = 30 * 100
SEQ_LEN = 10
D_MODEL = 128
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 手动文件配对（按你的实际文件配置）
FILE_PAIRS = {
    "SC4001E0-PSG.edf": "SC4001EC-Hypnogram.edf",
    "SC4002E0-PSG.edf": "SC4002EC-Hypnogram.edf",
    "SC4011E0-PSG.edf": "SC4011EH-Hypnogram.edf",
    "SC4012E0-PSG.edf": "SC4012EC-Hypnogram.edf",
    "SC4021E0-PSG.edf": "SC4021EH-Hypnogram.edf",
    "SC4022E0-PSG.edf": "SC4022EJ-Hypnogram.edf",
    "SC4031E0-PSG.edf": "SC4031EC-Hypnogram.edf",
    "SC4032E0-PSG.edf": "SC4032EP-Hypnogram.edf",
    "SC4041E0-PSG.edf": "SC4041EC-Hypnogram.edf",
    "SC4042E0-PSG.edf": "SC4042EC-Hypnogram.edf"
}


# 数据预处理
def load_and_preprocess():
    eeg_signals, sleep_labels = [], []
    for psg_file, ann_file in FILE_PAIRS.items():
        # 读取信号+标注文件
        raw = mne.io.read_raw_edf(os.path.join(DATA_PATH, psg_file), preload=True, verbose=False)
        ann = mne.read_annotations(os.path.join(DATA_PATH, ann_file))

        # 选EEG通道+滤波
        eeg_chs = [ch for ch in raw.info['ch_names'] if 'EEG' in ch]
        raw.pick([eeg_chs[0]] if len(eeg_chs) > 0 else [0])
        raw.filter(0.5, 30, fir_design='firwin', verbose=False)

        # 提取信号
        data, _ = raw[:]
        eeg_signals.append(data.squeeze())

        # 解析标签
        labels = []
        for desc in ann.description:
            if 'W' in desc:
                labels.append(0)
            elif '1' in desc:
                labels.append(1)
            elif '2' in desc:
                labels.append(2)
            elif '3' in desc or '4' in desc:
                labels.append(3)
            elif 'R' in desc:
                labels.append(4)
            else:
                labels.append(-1)
        sleep_labels.append(np.array(labels)[np.array(labels) != -1])

    # 窗口切分+归一化
    X, y = [], []
    for sig, lab in zip(eeg_signals, sleep_labels):
        n_win = min(len(sig) // WIN_SIZE, len(lab))
        X.append(np.reshape(sig[:n_win * WIN_SIZE], (n_win, WIN_SIZE)))
        y.append(lab[:n_win])
    X, y = np.concatenate(X), np.concatenate(y)
    X = StandardScaler().fit_transform(X)

    # 构建时序序列
    X_seq, y_seq = [], []
    for i in range(SEQ_LEN, len(X)):
        X_seq.append(X[i - SEQ_LEN:i, :])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


# 位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


# Transformer模型
class SleepTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(1, D_MODEL, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.pos_encoder = PositionalEncoding(D_MODEL)
        encoder_layers = nn.TransformerEncoderLayer(D_MODEL, 4, 512, 0.1, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layers, 2)
        self.fc = nn.Sequential(
            nn.Linear(D_MODEL, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(-1, 1, WIN_SIZE)
        x = self.embedding(x).squeeze(-1)
        x = x.reshape(batch_size, SEQ_LEN, D_MODEL).permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[-1, :, :])


# 数据加载
X_seq, y_seq = load_and_preprocess()
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

# 张量转换
X_train, X_val, X_test = map(lambda x: torch.tensor(x, dtype=torch.float32).to(DEVICE), [X_train, X_val, X_test])
y_train, y_val, y_test = map(lambda x: torch.tensor(x, dtype=torch.long).to(DEVICE), [y_train, y_val, y_test])

# 构建DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), BATCH_SIZE, shuffle=False)

# 模型初始化
model = SleepTransformer().to(DEVICE)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu()), y=y_train.cpu().numpy())
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)


# 训练函数
def train():
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                val_loss += criterion(model(batch_X), batch_y).item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | 训练损失: {train_loss / len(train_loader):.4f} | 验证损失: {val_loss / len(val_loader):.4f}")
    torch.save(model.state_dict(), "./transformer_model.pth")


# 评估函数
def evaluate():
    model.load_state_dict(torch.load("./transformer_model.pth"))
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            _, pred = torch.max(model(batch_X), 1)
            test_preds.extend(pred.cpu().numpy())
            test_true.extend(batch_y.cpu().numpy())

    label_names = ['W(清醒)', 'N1', 'N2', 'N3(深睡)', 'REM']
    print("\n========== Transformer模型分类报告 ==========")
    print(classification_report(test_true, test_preds, target_names=label_names, digits=4))

    cm = confusion_matrix(test_true, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('Transformer睡眠分期混淆矩阵')
    plt.tight_layout()
    plt.savefig("./transformer_confusion_matrix.png", dpi=300)


# 主函数
if __name__ == "__main__":
    print("加载数据...")
    load_and_preprocess()
    print("训练模型...")
    train()
    print("评估模型...")
    evaluate()
    print("模型落地完成")