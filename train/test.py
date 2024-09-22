import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 모델 및 토크나이저 초기화
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained('train/saved_model')

# CUDA 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def predict(text):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.sigmoid(logits).cpu().numpy()

    return predictions

while True:
    text = input('검증할 텍스트: ')

    # 예측 및 결과 출력
    predictions = predict(text)
    print(predictions)