"""
Model architecture that matches the saved checkpoint
"""
import torch
import torch.nn as nn

class CRNN_OLD(nn.Module):
    """Original CRNN architecture without BatchNorm in early layers"""
    def __init__(self, img_height, nn_classes):
        super(CRNN_OLD, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 0
            nn.ReLU(),                                              # 1
            nn.MaxPool2d(2, 2),                                     # 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1), # 3
            nn.ReLU(),                                              # 4
            nn.MaxPool2d(2, 2),                                     # 5
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 6
            nn.ReLU(),                                               # 7
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 8
            nn.ReLU(),                                               # 9
            nn.MaxPool2d((2, 1), (2, 1)),                            # 10
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 11
            nn.BatchNorm2d(512),                                     # 12
            nn.ReLU(),                                               # 13
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 14
            nn.BatchNorm2d(512),                                     # 15
            nn.ReLU(),                                               # 16
            nn.MaxPool2d((2, 1), (2, 1)),                            # 17
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 18
            nn.ReLU(),                                               # 19
            nn.MaxPool2d((2, 1), (2, 1))                             # 20
        )
        self.rnn = nn.LSTM(512, 512, num_layers=3, bidirectional=True, dropout=0.5)
        self.embedding = nn.Linear(512 * 2, nn_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.embedding(x)
        return x

if __name__ == "__main__":
    # Test loading
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from char_map import char_to_idx

    num_classes = len(char_to_idx) + 1
    model = CRNN_OLD(img_height=32, nn_classes=num_classes)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    print("✅ Successfully loaded checkpoint with CRNN_OLD architecture!")
    print(f"   Model has {num_classes} output classes")
