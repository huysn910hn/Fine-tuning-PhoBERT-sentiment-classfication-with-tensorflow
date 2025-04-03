## Fine-tuning-PhoBERT-sentiment-classfication-with-tensorflow
Mục tiêu: Phân tích cảm xúc học sinh, sinh viên: Hiểu rõ hơn về môi trường giáo dục

Data: Sử dụng dữ liệu của UIT

Nhãn: tiêu cực: 0, trung lập: 1, tích cực: 2


## Cấu trúc thư mục


     ├── main.ipynb                            
     ├── data.csv                             
     ├── 📂 app/                                 
     │   ├── Dockerfile                 
     │   ├── phobert_sentiment_model5.keras     
     │   ├── requirements.txt
     │   ├── server.py
     └── README.md

## Cài đặt
1. Cài đặt các framework
     ```bash
     pip install -r requirements.txt

1. Build docker:
     ```bash
     docker build -t name .
     
2. Run docker:
     ```bash
     docker run -p 8000:8000 name

3. Truy cập vào localhost:
     ```bash
     http://localhost:8000/docs
