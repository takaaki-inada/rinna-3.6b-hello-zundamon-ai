# ChatVRM for hello-zundamon-ai

This repository is based on [pixiv/ChatVRM](https://github.com/pixiv/ChatVRM).

---
ChatVRMはブラウザで簡単に3Dキャラクターと会話ができるデモアプリケーションです。

VRMファイルをインポートしてキャラクターに合わせた声の調整や、感情表現を含んだ返答文の生成などを行うことができます。

ChatVRMの各機能は主に以下の技術を使用しています。

- ユーザーの音声の認識
    - [Web Speech API(SpeechRecognition)](https://developer.mozilla.org/ja/docs/Web/API/SpeechRecognition)
- 読み上げ音声の生成
    - [Koemotion/Koeiromap API](https://koemotion.rinna.co.jp/)
- 3Dキャラクターの表示
    - [@pixiv/three-vrm](https://github.com/pixiv/three-vrm)


## 実行
ローカル環境で実行する場合はこのリポジトリをクローンするか、ダウンロードしてください。

必要なパッケージをインストールしてください。
```bash
npm install
```

パッケージのインストールが完了した後、以下のコマンドで開発用のWebサーバーを起動します。
```bash
npm run dev
```

実行後、以下のURLにアクセスして動作を確認して下さい。

[http://localhost:3000](http://localhost:3000)


---

## Koeiromap API
ChatVRMでは返答文の音声読み上げにKoemotionのKoeiromap APIを使用しています。

Koeiromap APIの仕様や利用規約については以下のリンクや公式サイトをご確認ください。

- [https://koemotion.rinna.co.jp/](https://koemotion.rinna.co.jp/)

## Zundamon VRM
[ずんだもん（人型）公式MMDモデル、VRM、VRChatアバター - 東北ずん子ショップ【公式】 - BOOTH](https://booth.pm/ja/items/3733351)
