# FLW-Net-onnx2tf-sample
Low-Light Image Enhancementモデルである[hitzhangyu/FLW-Net](https://github.com/hitzhangyu/FLW-Net)のPythonでのONNX、TFLite推論サンプルです。<br>
ONNX、TFLiteに変換したモデルも同梱しています。変換自体を試したい方は[FLW_Net_onnx2tf.ipynb](FLW_Net_onnx2tf.ipynb)を使用ください。<br>
TFLiteへの変換には[PINTO0309/onnx2tf](https://github.com/PINTO0309/onnx2tf)を使用しています。

https://user-images.githubusercontent.com/37477845/231034876-276ed2a0-bdcd-406d-b336-198748161db5.mp4

# Requirement 
* OpenCV 4.5.3.56 or later
* onnxruntime 1.13.0 or later
* tensorflow 2.9.1 or later

# Demo
デモの実行方法は以下です。
```bash
python demo_onnx.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/FLW-Net_320x240.onnx

```bash
python demo_tflite.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/saved_model_320x240/FLW-Net_320x240_float16.tflite
* --num_threads<br>
推論時に使用するスレッド数<br>
デフォルト：1

# Reference
* [hitzhangyu/FLW-Net](https://github.com/hitzhangyu/FLW-Net)
* [Dovyski/cvui](https://github.com/Dovyski/cvui)
* [PINTO0309/onnx2tf](https://github.com/PINTO0309/onnx2tf)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
FLW-Net-onnx2tf-sample is under [MIT License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[雨イメージ　夜の道路を走る車](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002161702_00000)を使用しています。
