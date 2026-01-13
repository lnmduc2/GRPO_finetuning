Cài vllm theo doc: 
```
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

Chạy script setup
```
bash setup.sh
```

Cuối cùng
```
uv run train.py
```

Các checkpoint sẽ được lưu ở folder `outputs`

