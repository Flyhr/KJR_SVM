import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# 加载预先训练好的 SVM 模型流水线（确保 svm_pipeline.pkl 文件存在）
model = joblib.load("svm_pipeline.pkl")


@app.route('/api/healthData/predict', methods=['GET'])
def predict():
    try:
        # 从请求参数中获取所需数据，确保名称与训练时的字段一致
        new_data = {
            "weight": float(request.args.get('weight', 70.5)),
            "height": float(request.args.get('height', 175.0)),
            "gender": int(request.args.get('gender', 1)),  # 男为1，女为0
            "age": int(request.args.get('age', 35)),
            "smoking": int(request.args.get('smoking', 1)),  # 吸烟：1 是，0 否
            "temp": float(request.args.get('temp', 36.8)),
            "temp_cut": float(request.args.get('temp_cut', 36.5)),
            "pulse": int(request.args.get('pulse', 75)),
            "sbp": int(request.args.get('sbp', 120)),
            "dbp": int(request.args.get('dbp', 80)),
            "swelling": int(request.args.get('swelling', 0)),  # 0 否，1 是
            "knee": int(request.args.get('knee', 15)),
            "step": int(request.args.get('step', 5000)),
            "decri": request.args.get('decri', "轻微疼痛，无明显不适")
        }

        # 将数据转换为 DataFrame 格式，因为模型训练时使用了 DataFrame 作为输入
        data_df = pd.DataFrame([new_data])
        # 对新数据调用模型进行预测，返回值一般为 0、1 或 2
        predicted_pain = int(model.predict(data_df)[0])

        # 构造返回结果，只包含预测值
        result = {
            "predictedPain": predicted_pain
        }
        print("预测结果：", result)
        return jsonify({
            "code": 200,
            "data": result,
            "msg": "预测成功"
        })
    except Exception as e:
        errMsg = "预测错误：" + str(e)
        print(errMsg)
        return jsonify({
            "code": 500,
            "msg": errMsg
        })


if __name__ == '__main__':
    # 设置监听所有网卡，端口为 4672
    app.run(host="0.0.0.0", port=4672)
