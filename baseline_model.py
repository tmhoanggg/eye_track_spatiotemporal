import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CNN_GRU(nn.Module):
    """
        A baseline eye tracking which uses CNN + GRU to predict the pupil center coordinate
    """
    def __init__(self, args):
        super().__init__() 
        self.args = args
        self.conv1 = nn.Conv2d(args.n_time_bins, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.gru = nn.GRU(input_size=36192, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 2)


    def forward(self, x):
        # input is of shape (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size*seq_len, channels, height, width)
        # permute height and width
        x = x.permute(0, 1, 3, 2)

        x= self.conv1(x)
        x= torch.relu(x)
        x= self.conv2(x)
        x= torch.relu(x)
        x= self.conv3(x)
        x= torch.relu(x)
        x= self.pool(x)

        x = x.view(batch_size, seq_len, -1)
        x, _ = self.gru(x)
        # output shape of x is (batch_size, seq_len, hidden_size)

        x = self.fc(x)
        # output is of shape (batch_size, seq_len, 2)
        return x
    


class Resnet_GRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load ResNet18 pretrained, bỏ layer cuối (fc)
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Loại bỏ FC layer

        # GRU xử lý chuỗi thời gian
        self.gru = nn.GRU(input_size=512, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, 2)  # Bidirectional nên nhân đôi hidden_size

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)

        x = self.cnn(x)  # Feature shape: (batch_size*seq_len, 512, 1, 1)
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 512)

        x, _ = self.gru(x)  # (batch_size, seq_len, 256)
        x = self.fc(x)  # (batch_size, seq_len, 2)
        return x

class Resnet_LSTM(nn.Module):
    def __init__(self, args):
        super(Resnet_LSTM, self).__init__()
        self.args = args
        
        # Load ResNet18 pretrained và loại bỏ lớp fully-connected cuối cùng
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Kết quả: (batch_size*seq_len, 512, 1, 1)
        
        # LSTM để xử lý chuỗi thời gian. Sử dụng LSTM 2 tầng, bidirectional với hidden_size=128
        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=2, 
                            batch_first=True, bidirectional=True)
        
        # Fully-connected layer để chuyển đổi đầu ra của LSTM thành tọa độ (x, y)
        self.fc = nn.Linear(128 * 2, 2)  # bidirectional nên nhân đôi hidden_size

    def forward(self, x):
        # x có shape: (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        
        # Gộp batch_size và seq_len để đưa vào CNN
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.cnn(x)  # Shape: (batch_size*seq_len, 512, 1, 1)
        x = x.view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, 512)
        
        # Xử lý chuỗi với LSTM
        x, _ = self.lstm(x)  # Shape: (batch_size, seq_len, 256)
        
        # Dự đoán tọa độ (x, y) cho mỗi frame
        x = self.fc(x)  # Shape: (batch_size, seq_len, 2)
        return x


class ViT_GRU(nn.Module):
    def __init__(self, args):
        super(ViT_GRU, self).__init__()
        self.args = args
        
        # Lấy mô hình Vision Transformer từ torchvision (ViT-B/16)
        self.vit = models.vit_b_16(pretrained=True)
        
        # Lấy kích thước đầu ra của ViT
        feature_dim = self.vit.heads[0].in_features  # ✅ Cách đúng để lấy feature_dim
        
        # Bỏ classifier để lấy đặc trưng thô
        self.vit.heads = nn.Identity()

        # GRU để xử lý chuỗi thời gian
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=128, num_layers=2, 
                          batch_first=True, bidirectional=True)
        
        # Fully connected để chuyển GRU output thành tọa độ (x, y)
        self.fc = nn.Linear(128 * 2, 2)  # Bidirectional nên nhân đôi hidden_size
        
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
    
        # Resize input về 224x224
        x = x.view(batch_size * seq_len, channels, height, width)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    
        # Trích xuất đặc trưng từ ViT
        features = self.vit(x)
        
        # Reshape lại thành chuỗi thời gian
        features = features.view(batch_size, seq_len, -1)
        
        # Đưa vào GRU
        gru_out, _ = self.gru(features)
        
        # Dự đoán tọa độ
        out = self.fc(gru_out)
        return out

class SwinT_GRU(nn.Module):
    def __init__(self, args):
        super(SwinT_GRU, self).__init__()
        self.args = args
        
        # Load mô hình Swin Transformer từ torchvision (Swin-Tiny)
        self.swin = models.swin_t(pretrained=True)
        
        # Lấy kích thước đầu ra của Swin Transformer (thường là 768)
        feature_dim = self.swin.head.in_features  # Swin-T output features
        
        # Loại bỏ classifier cuối của Swin Transformer
        self.swin.head = nn.Identity()
        
        # GRU để xử lý chuỗi thời gian
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=128, num_layers=2, 
                          batch_first=True, bidirectional=True)
        
        # Fully connected để chuyển GRU output thành tọa độ (x, y)
        self.fc = nn.Linear(128 * 2, 2)  # Bidirectional nên nhân đôi hidden_size
        
    def forward(self, x):
        """
        Đầu vào x có shape: (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Gộp batch_size và seq_len lại để đưa vào Swin Transformer
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Trích xuất đặc trưng không gian từ Swin Transformer
        features = self.swin(x)  # Shape: (batch_size*seq_len, feature_dim)
        
        # Reshape lại thành chuỗi theo thời gian: (batch_size, seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)
        
        # Xử lý chuỗi với GRU
        gru_out, _ = self.gru(features)  # Output: (batch_size, seq_len, 256)
        
        # Dự đoán tọa độ (x, y) cho từng frame
        out = self.fc(gru_out)  # Shape: (batch_size, seq_len, 2)
        return out

class EfficientNet_GRU(nn.Module):
    def __init__(self, args):
        super(EfficientNet_GRU, self).__init__()
        self.args = args
        
        # Load EfficientNet-B0 pretrained
        self.effnet = models.efficientnet_b0(pretrained=True)
        
        # Lấy feature size từ EfficientNet
        feature_dim = self.effnet.classifier[1].in_features
        
        # Bỏ fully-connected layer cuối
        self.effnet.classifier = nn.Identity()
        
        # GRU để xử lý chuỗi thời gian
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=128, num_layers=2, 
                          batch_first=True, bidirectional=True)
        
        # Fully connected để chuyển GRU output thành tọa độ (x, y)
        self.fc = nn.Linear(128 * 2, 2)  # Bidirectional nên nhân đôi hidden_size
        
    def forward(self, x):
        """
        Đầu vào x có shape: (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Gộp batch_size và seq_len lại để đưa vào EfficientNet
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Trích xuất đặc trưng không gian từ EfficientNet
        features = self.effnet(x)  # Shape: (batch_size*seq_len, feature_dim)
        
        # Reshape lại thành chuỗi theo thời gian: (batch_size, seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)
        
        # Xử lý chuỗi với GRU
        gru_out, _ = self.gru(features)  # Output: (batch_size, seq_len, 256)
        
        # Dự đoán tọa độ (x, y) cho từng frame
        out = self.fc(gru_out)  # Shape: (batch_size, seq_len, 2)
        return out

class EfficientNetB3_GRU(nn.Module):
    def __init__(self, args):
        super(EfficientNetB3_GRU, self).__init__()
        self.args = args
        
        # Load EfficientNet-B3 pretrained
        self.effnet = models.efficientnet_b3(pretrained=True)
        
        # Lấy feature size từ EfficientNet
        feature_dim = self.effnet.classifier[1].in_features
        
        # Bỏ fully-connected layer cuối
        self.effnet.classifier = nn.Identity()
        
        # GRU để xử lý chuỗi thời gian
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=128, num_layers=2, 
                          batch_first=True, bidirectional=True)
        
        # Fully connected để chuyển GRU output thành tọa độ (x, y)
        self.fc = nn.Linear(128 * 2, 2)  # Bidirectional nên nhân đôi hidden_size
        
    def forward(self, x):
        """
        Đầu vào x có shape: (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Gộp batch_size và seq_len lại để đưa vào EfficientNet
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Trích xuất đặc trưng không gian từ EfficientNet
        features = self.effnet(x)  # Shape: (batch_size*seq_len, feature_dim)
        
        # Reshape lại thành chuỗi theo thời gian: (batch_size, seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)
        
        # Xử lý chuỗi với GRU
        gru_out, _ = self.gru(features)  # Output: (batch_size, seq_len, 256)
        
        # Dự đoán tọa độ (x, y) cho từng frame
        out = self.fc(gru_out)  # Shape: (batch_size, seq_len, 2)
        return out

class EfficientNet_LSTM(nn.Module):
    def __init__(self, args):
        super(EfficientNet_LSTM, self).__init__()
        self.args = args
        
        # Load EfficientNet-B0 pretrained từ torchvision
        self.effnet = models.efficientnet_b0(pretrained=True)
        
        # Lấy số lượng features từ classifier của EfficientNet
        feature_dim = self.effnet.classifier[1].in_features
        
        # Bỏ classifier cuối cùng để chỉ lấy feature
        self.effnet.classifier = nn.Identity()
        
        # LSTM thay vì GRU
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=128, num_layers=2, 
                            batch_first=True, bidirectional=True)
        
        # Fully connected để dự đoán tọa độ (x, y)
        self.fc = nn.Linear(128 * 2, 2)  # Nhân đôi hidden_size vì bidirectional
        
    def forward(self, x):
        """
        Đầu vào x có dạng: (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Gộp batch_size và seq_len để đưa vào EfficientNet
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Trích xuất đặc trưng từ EfficientNet
        features = self.effnet(x)  # Shape: (batch_size*seq_len, feature_dim)
        
        # Reshape lại thành dạng chuỗi thời gian
        features = features.view(batch_size, seq_len, -1)
        
        # Xử lý bằng LSTM
        lstm_out, _ = self.lstm(features)  # Output: (batch_size, seq_len, 256)
        
        # Dự đoán tọa độ mắt (x, y)
        out = self.fc(lstm_out)  # Shape: (batch_size, seq_len, 2)
        
        return out



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        """
        x có shape: (batch_size, seq_len, hidden_size)
        """
        attn_weights = torch.softmax(self.attn(x), dim=1)  
        context = attn_weights * x  # Giữ seq_len
        return context  # Shape: (batch_size, seq_len, hidden_size)

class EfficientNet_GRU_Attention(nn.Module):
    def __init__(self, args):
        super(EfficientNet_GRU_Attention, self).__init__()
        self.args = args
        
        # Load EfficientNet-B0 pretrained
        self.effnet = models.efficientnet_b0(pretrained=True)
        
        # Lấy feature size từ EfficientNet
        feature_dim = self.effnet.classifier[1].in_features
        
        # Bỏ fully-connected layer cuối
        self.effnet.classifier = nn.Identity()
        
        # GRU để xử lý chuỗi thời gian
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=128, num_layers=2, 
                          batch_first=True, bidirectional=True)
        
        # Thêm Attention để trọng số hóa thông tin từ GRU
        self.attention = Attention(128 * 2)  # GRU bidirectional nên hidden_size x2
        
        # Fully connected để chuyển GRU output thành tọa độ (x, y)
        self.fc = nn.Linear(128 * 2, 2) 
        
    def forward(self, x):
        """
        Đầu vào x có shape: (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Gộp batch_size và seq_len lại để đưa vào EfficientNet
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Trích xuất đặc trưng không gian từ EfficientNet
        features = self.effnet(x)  # Shape: (batch_size*seq_len, feature_dim)
        
        # Reshape lại thành chuỗi theo thời gian: (batch_size, seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)
        
        # Xử lý chuỗi với GRU
        gru_out, _ = self.gru(features)  # Shape: (batch_size, seq_len, 256)
        
        # Áp dụng Attention để lấy trọng số quan trọng
        context = self.attention(gru_out)  # Shape: (batch_size, 256)
        
        # Dự đoán tọa độ (x, y)
        out = self.fc(context)  # Shape: (batch_size, 2)


        return out

class EfficientNet_BiGRU_S6(nn.Module):
    def __init__(self, args):
        super(EfficientNet_BiGRU_S6, self).__init__()
        self.args = args
        
        # Load EfficientNet-B0 pretrained
        self.effnet = models.efficientnet_b0(pretrained=True)
        
        # Lấy feature size từ EfficientNet
        feature_dim = self.effnet.classifier[1].in_features
        
        # Bỏ fully-connected layer cuối
        self.effnet.classifier = nn.Identity()
        
        # GRU xử lý chuỗi thời gian (Bidirectional)
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=128, num_layers=2, 
                          batch_first=True, bidirectional=True)
        
        # Thay Attention bằng Mamba (S6-based) để xử lý thông tin chuỗi dài
        self.s6 = Mamba(d_model=256, d_state=64, d_conv=4, expand=2)  # 256 do GRU bidirectional
        
        # Fully connected để chuyển output thành tọa độ (x, y)
        self.fc = nn.Linear(256, 2)  # 256 do BiGRU đầu ra (128 x2)

        # Layer Normalization giúp ổn định đầu vào cho S6
        self.norm = nn.LayerNorm(256)
        
    def forward(self, x):
        """
        Đầu vào x có shape: (batch_size, seq_len, channels, height, width)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Gộp batch_size và seq_len để đưa vào EfficientNet
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Trích xuất đặc trưng không gian từ EfficientNet
        features = self.effnet(x)  # Shape: (batch_size*seq_len, feature_dim)
        
        # Reshape lại thành chuỗi theo thời gian: (batch_size, seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)
        
        # Xử lý chuỗi với BiGRU
        gru_out, _ = self.gru(features)  # Shape: (batch_size, seq_len, 256)
        
        # Chuẩn hóa đầu vào cho S6
        gru_out = self.norm(gru_out)
        
        # Áp dụng S6 (Mamba) để xử lý thông tin chuỗi dài
        s6_out = self.s6(gru_out)  # Shape: (batch_size, seq_len, 256)
        
        # Residual Connection: Giữ lại thông tin từ BiGRU
        s6_out = s6_out + gru_out  # Shape: (batch_size, seq_len, 256)
        
        # Lấy thông tin cuối cùng (tại thời điểm seq_len) để dự đoán
        final_out = s6_out[:, -1, :]  # Shape: (batch_size, 256)
        
        # Dự đoán tọa độ (x, y)
        out = self.fc(final_out)  # Shape: (batch_size, 2)
        
        return out

class ConvNeXt_BiGRU(nn.Module):
    def __init__(self, args):
        super(ConvNeXt_BiGRU, self).__init__()
        self.args = args
        
        # 🔥 Dùng ConvNeXt làm backbone thay vì EfficientNet
        self.backbone = models.convnext_base(weights="IMAGENET1K_V1")
        feature_dim = self.backbone.classifier[2].in_features
        
        # Bỏ fully-connected cuối
        self.backbone.classifier = nn.Identity()
        
        # GRU xử lý chuỗi thời gian (Bidirectional)
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=128, num_layers=2, 
                          batch_first=True, bidirectional=True)
        
        # Fully connected để chuyển output thành tọa độ (x, y)
        self.fc = nn.Linear(256, 2)  # 256 do BiGRU đầu ra (128 x2)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape

        # Gộp batch_size và seq_len để đưa vào ConvNeXt
        x = x.view(batch_size * seq_len, channels, height, width)

        # 🔥 Trích xuất đặc trưng bằng ConvNeXt
        features = self.backbone(x)  # Shape: (batch_size*seq_len, feature_dim)

        # Reshape lại thành chuỗi theo thời gian: (batch_size, seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)

        # Xử lý chuỗi với BiGRU
        gru_out, _ = self.gru(features)  # Shape: (batch_size, seq_len, 256)

        # Lấy thông tin cuối cùng (tại thời điểm seq_len) để dự đoán
        final_out = gru_out[:, -1, :]  # Shape: (batch_size, 256)

        # Dự đoán tọa độ (x, y)
        out = self.fc(final_out)  # Shape: (batch_size, 2)

        return out
