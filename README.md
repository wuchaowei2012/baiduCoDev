
安装环境指令
```shell
conda create --name baidu python==3.10.12  
conda activate baidu
pip install -r requirement.txt

python train.py data/demo
````


生成的模型会保存在 model下面，我们现在是使用tf-serving docker直接拉起的
指令如下
```
docker run --gpus all  -p 9501:8501 --name baidu  -v /data/fred/searching_baidu/model:/models/baidu -e MODEL_NAME=baidu -t tensorflow/serving:2.8.0-gpu
```

入参非常简单，测试服务代码如下
```shell
curl --location 'http://localhost:9501/v1/models/baidu:preict' --header 'Content-Type: application/json' --data '{
    "instances": [["思齐贝比SIKKIBABY"]]
```