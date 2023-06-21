FROM python:3.9

COPY . .

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install --no-cache-dir -r requirements_inference.txt
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

CMD ["python", "app.py"]
