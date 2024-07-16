繁 ｜ [简](./README_SC.md) ｜ [EN](./README.md)

<p align="center"><img src="assets/icon.jpg" width="150"/></p>

<p align="center">
原創性：
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
&nbsp
創新性：
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
&nbsp
挑戰性：
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
</p>

<p align="center">
🛠️ <a href="#運作原理">運作原理</a>
｜
📁 <a href="#文件結構">文件結構</a>
｜
🖥️ <a href="#使用說明">使用說明</a>
｜
👀 <a href="#結果示例">結果示例</a>
｜
📣 <a href="#常見報錯">常見報錯</a>
｜
🙋🏻‍♂️ <a href="#常見問題">常見問題</a>
</p>


# 連登語言模型 LiLM

受 [林亦](https://www.youtube.com/@lyi) 的 [bilibot 項目](https://github.com/linyiLYi/bilibot/tree/main) 和 [影片](https://www.youtube.com/watch?v=52clfKcM4M4&t=1s) 所啟發，本實驗性項目使用帶有獨特語言風格的 [LIHKG 討論區](https://lihkg.com) 用戶回應作微調訓練，創建了這個廣東話貼文回應生成語言模型。

在平衡運算成本和 [基礎模型的中文能力](https://github.com/jeinlee1991/chinese-llm-benchmark) 後，本實驗性項目選用的開源基礎模型為 [Qwen/Qwen1.5-32B-Chat](https://huggingface.co/Qwen/Qwen1.5-32B-Chat)，具有320億參數，借助 Apple Silicon 平台 AI 專用框架 [MLX](https://github.com/ml-explore/mlx) 的 [MLX-LM LoRA 微調示例](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#fine-tune)，利用 [LoRA 算法](https://arxiv.org/abs/2106.09685) 在 M3 Max 128GB 和 M2 Ultra 192GB 上對基礎模型進行微調。

微調後的模型在廣東話語言能力上有顯著提升，其語氣和風格亦深受一眾 [連登](https://zh.wikipedia.org/zh-hk/LIHKG討論區) 用戶的薰陶，詳情可參見 [結果示例](#結果示例)。模型已上載至 Hugging Face: [alphrc/lilm](https://huggingface.co/alphrc/lilm/tree/main)。

若想了解更多關於人工智能的資訊，並在未來查看更多創新有趣的項目，請關注 [alphrc](https://github.com/alphrc)。


### 項目動機
- 本項目旨在展示大型語言模型基於廣東話口語數據和獨特語言風格的語言風格模仿能力，主要用於普及教學、學術研究和技術展示，因此內容會比較詳細

### 使用限制
- 模型訓練基於公開數據，雖已努力清理敏感內容，但仍可能包含基於訓練內容的偏見，使用時應避免生成不當內容
- 生成的文本反映特定社群文化，使用前應了解相關背景
- 在實際應用前進行充分測試，避免在敏感或爭議性場合使用，並設立監控機制以防生成不當內容

### 備注
- 本項目所有代碼均為自行編寫，同時亦鼓勵開源社區成員對項目進行審查，提供反饋和建議，以及直接參與項目的改進
- 本項目的性質為第三方訓練框架和模型的使用和實踐，主要挑戰在於系統配置、數據抓取、數據工程、反覆試錯和漫長的等待
- 本項目整理了部分配置信息和內容在 `.env` 文件中，以便使用者根據個人或組織的具體需求自行調整，確保靈活性和適用性，其格式已放置在 `.env.template`，使用時把檔案名稱改為 `.env` 便可


## 運作原理

### 微調
大型預訓練語言模型 ([pretrained language model](https://www.kaggle.com/code/vad13irt/language-model-pre-training))，擁有基本和通用的人類語言對答能力。透過特定文字數據對模型進行微調 ([fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)))，能讓它在這些數據上進一步學習，增加其對數據的模仿能力，如語氣、風格、資訊和用詞等等。要注意，使用特定數據進行微調並不能讓模型從零開始得到語言能力，只能在其本來預訓練而獲得的語言能力上，加深其對局部文字資訊和規律的理解。

### 數據集
本項目在 [LIHKG 討論區](https://lihkg.com) 進行公開數據的大規模抓取，並透過對原數據的過濾和處理以產生數據集用作微調。為提高數據質量，其過濾基準為：

- 貼文的第一則回應不是作者本人，確保回應所基於的資訊的完整性
- 回應反應正面，確保該回應符合討論區主流意見
- 回應的總反應數目不少於 20 以降低噪聲
- 不是在回覆其他回應
- 不是作者自己的回應
- 不包含任何外部連結和鑲嵌
- 不包含敏感字眼
- 總字數加上系統信息不超過 2048

其後以這些回應結合相應貼文的標題、內容和分類，加上系統信息 ([system message](https://promptmetheus.com/resources/llm-knowledge-base/system-message))，轉換成 MLX-LM LoRA 微調示例中所要求的 [格式](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data)，隨機排列後生成總數據集。總數據集會以 80 : 10 : 10 的比例分割為訓練集 (training set)、驗證集 (validation set) 和測試集 (testing set)，當中測試集的貼文完全沒有在訓練集或驗證集中出現過，以驗證泛化 ([generalization](https://towardsdatascience.com/generalization-in-ai-systems-79c5b6347f2c)) 和避免過擬合 ([overfitting](https://en.wikipedia.org/wiki/Overfitting))。

最終版本的訓練集中包含了約 60000 個貼文中符合基準的 27792 項數據，驗證集和測試集則各包含 3474 項數據。

### 基礎模型
開源基礎模型 [Qwen/Qwen1.5-32B-Chat](https://huggingface.co/Qwen/Qwen1.5-32B-Chat) 具有320億參數，精度為 BF16。系統在首次使用 MLX-LM 模組運行的時候，若檢測到 `~/.cache` 中沒有下載模型，便會自動從 Hugging Face 上下載模型到 `~/.cache/huggingface/hub/model--Qwen--Qwen1.5-32B-Chat`，用家不需要手動預先下載。模型的大小約為 65GB，分為若干個板塊下載，若下載過程受到中斷，模型會在下次下載的時候自動搜集已下載的板塊以繼續下載，不用擔心需要重新下載。

### LoRA
在傳統的訓練和微調方法中，需要同時把模型中一些大型矩陣的所有參數進行調整，對電腦內存和計算量有很高的要求。與傳統方法相比，[LoRA (Low Rank Adaption)](https://arxiv.org/abs/2106.09685) 利用兩個小型矩陣去估算模型中的大型矩陣的變化，大幅降低參數的數量，令模型能在較低內存的設備上也能進行輕量化微調，大幅降低了訓練所需的時間。在實際執行中，原來模型的總參數量為 32.5B，而在基礎模型的所有 63 層 attention layer 上使用 LoRA 後，可學習的參數量為 8.3M，僅為本來的 0.026%。

使用 MLX-LM LoRA 去微調模型並不會改變模型本來的參數，而是另外生成 adapters 與模型配合使用。在微調的過程中，MLX-LM 會自動在當前工作目錄 (current working directory) 生成 `adapters/` 文件夾，並把 adapter 的 checkpoints 以 `.safetensors` 格式保存下來，每個 checkpoint 的大小約為 33.6MB，這些 checkpoints 能在之後繼續使用作微調。

### 梯度檢查點
梯度檢查點 (gradient checkpointing) 是一種用於訓練大型神經網絡時節省記憶體的技術。在神經網絡訓練過程中，為了進行有效的反向傳播 ([backpropagation](https://brilliant.org/wiki/backpropagation/#:~:text=Backpropagation%2C%20short%20for%20%22backward%20propagation,to%20the%20neural%20network's%20weights.))，通常需要保存中間層的輸出結果以用於計算梯度 (gradient)。然而，這會消耗大量的記憶體，尤其是在處理深層網絡時。梯度檢查點的方法是在訓練過程中不保存所有中間層的輸出，而只保存一些關鍵層的輸出。當需要進行梯度計算時，使用這些保存的關鍵點重建丟失的中間數據。這樣可以在保證訓練效果的同時，大幅減少記憶體的使用。

### 模型融合
微調完畢後，MLX-LM 能把 adapter 和原有的模型混合起來，在當前工作目錄中的 `model/lilm` 文件夾生成一個完整的模型，大小約為 65GB。之後便能直接透過此文件夾的路徑來使用此模型，不用再將原模型和 adapter 一起來使用。


## 文件結構
- `src/` : Python 代碼
    - `data.py` : 多執行緒代理數據抓取、格式化和初步處理 (需要 proxy 來運行)
    - `dataset.py` : 數據處理、變換和過濾
    - `run.py` : LiLM 模型封裝和基本用戶介面
- `data/` : 數據抓取得到的原數據，以 `.csv` 儲存
- `dataset/` : 處理後得到的訓練數據，根據格式分為 `completion/` 和 `chat/`
- `adapters/` : 儲存 `mlx_lm.lora` 自動生成的 adapters 和 configuration
- `adapters-llama3-70b/`: Llama3-70B 的 adapters
- `model/lilm` : 由基礎模型和 adapter 融合而成的融合模型，需運行以下 shell script 生成
- `demo/` : 範例數據，供 `run.py` 使用


## 使用說明

### 設備需求
因本項目使用 Apple 專有的 MLX 框架，所以只能在配有 Apple Silicon Chips (M1或以上) 的機器上的 MacOS 系統運行。本地機器需要約 75GB RAM 以順暢地進行推理，並需要約 122GB RAM 以順暢地進行微調。

### 環境配置
運行以下 shell script，利用 [Anaconda](https://www.anaconda.com) 建立和配置環境，並根據 `requirements.txt` 下載所有所需的 dependency。
```bash
conda create -n lilm python=3.9
conda activate lilm
pip install -r requirements.txt 
```

### 監測系統資源用量 (非必要)
利用 `asitop` 模組，透過圖像介面實時監測電腦資源用量，如 CPU、GPU 和 RAM 等，以確保程序在正常運行。

```bash
sudo asitop
```

### 利用基礎模型進行推理
首次運行會自動下載模型，`--model` 可以使用模型在 Hugging Face 上的全稱或其所在的路徑，
```bash
mlx_lm.generate \
    --model Qwen/Qwen1.5-32B-Chat \
    --prompt "咩係連登？"
```

### 微調
在 `dataset/chat` 預備數據集 `train.jsonl` 和 `valid.jsonl` 後，從頭開始微調模型，並生成 `adapters/` 文件夾。
```bash
mlx_lm.lora \
    --model Qwen/Qwen1.5-32B-Chat \
    --train \
    --data dataset/chat \
    --iters 600 \
    --grad-checkpoint
```

### 繼續微調
使用現有 adapter 繼續進行微調，`--resume-adapter-file` 必須為 `.safetensors` 文件。
```bash
mlx_lm.lora \
    --model Qwen/Qwen1.5-32B-Chat \
    --resume-adapter-file adapters/adapters.safetensors
    --train \
    --data dataset/chat \
    --iters 600 \
    --grad-checkpoint
```
🚨 請注意，你很可能會遇到 [此報錯](#錯誤-1)。

### 配合 adapter 進行推理
使用基礎模型配合 adapter 進行生成，adapter 必須為 `.safetensors` 文件。
```bash
mlx_lm.generate \
    --model Qwen/Qwen1.5-32B-Chat \
    --adapter-path adapters/adapters.safetensors \
    --prompt "咩係連登？"
```

### 融合基礎模型和 adapter
`adapters/` 中最新的 checkpoint `adapters.safetensors` 會被自動選取來進行融合，並把融合後的模型放置在 `model/lilm`。
```bash
mlx_lm.fuse \
    --model Qwen/Qwen1.5-32B-Chat \
    --adapter-path adapters \
    --save-path model/lilm
```

### 利用融合模型進行推理
`--model` 中使用融合模型的路徑。

```bash
mlx_lm.generate \
    --model model/lilm \
    --prompt "咩係連登？"
```

### 量化模型 (非必要)
利用量化 ([quantization](https://blog.csdn.net/jinzhuojun/article/details/106955059)) 降低模型參數精度，壓縮模型大小，加速推理和減少內存佔用空間。以下的 `--hf-path` 跟之前一樣，可以是 Hugging Face 上模型的完整名字，或者是模型的路徑，而 `--mlx-path` 則是壓縮後的模型所存放的路徑。然而實測後中量化模型準確度會大幅下降，且量化模型不能使用 Hugging Face 的 Transformer 運行。

```bash
mlx_lm.convert \
    --hf-path model/lilm \
    --mlx-path model/lilm-4Bit \
    -q
```

### 運行 LiLM
使用 `src/run.py` 來運行已融合的模型，可選擇 `interactive` 模式來輸入貼文連結作回應。

```bash
python src/run.py
```


## 結果示例

LiLM 比起基礎模型在廣東話語言能力上有顯著提升，其語言風格亦受到連登討論區影響。以下內容僅作示例用途，可能具有冒犯性，敏感字眼會以「X」顯示。

### 示例一 

**提示 ([原文](https://lihkg.com/thread/3699748/page/1))：**
> 類別：「創意台」<br><br>
> 標題：「有冇人覺得大廈看更根本冇X用？」<br><br>
> 內容：「唔係阿叔就係阿嬸，毫無阻嚇作用，遇到有人打劫都係匿鳩埋」<img src="https://lihkg.com/assets/faces/normal/dead.gif"/>

**使用 LiLM 作回應：**
> 我哋呢啲大廈都係咁，有時真係想問，佢哋係咪真係用來作擺設？ <img style="vertical-align: middle;" src="https://lihkg.com/assets/faces/normal/like.gif" />

或

> 有X用咩？大廈看更係冇用既XX，冇阻吓作用，冇得救 <img src="https://lihkg.com/assets/faces/normal/dead.gif"/>

**使用原有模型作回應：**
> <details>
> <summary>展開完整內容</summary><br>
> 在創意台的背景下，這個主題引起了討論。有些人可能會同意你的看法，認為大廈的看更可能並沒有提供足夠的安全感。他們可能會舉例，像你提到的，看更可能是年長者，可能在身體素質或反應能力上有所限制，面對突發情況時可能無法有效阻止犯罪。<br>
> 然而，也有人會強調看更的重要性。他們可能會指出，即使看更不一定能 physically 阻止犯罪，他們的存在本身就是一種威慑，因為他們能及時發現異常並通報。此外，看更還負責監控 CCTV、管理出入和確保大廈的基本安全，這些都是不容忽視的責任。<br><br>
> 另外，我們也可以從另一個角度來看這個問題：是否應該期待看更成為專業的治安維護者？或者我們需要更全面的治安措施，如加強警方巡邏、提升社區警覺性，甚至利用先進技術如 AI 監控等？<br><br>
> 總之，這個話題涉及對安全需求的理解、對社區角色的定義，以及如何最有效地分配資源以保障大家的安全。每個人可能都有自己獨特的觀點，這就是創意台的魅力之一，讓大家可以自由交流不同的想法。
> </details>

**使用原有模型加上系統提示作回應：**
> <details>
> <summary>展開完整內容</summary><br>
> 有冇搞錯？你以為大廈看更是超人定蝙蝠俠呀？
>
> 得閒啦你，大廈看更起码都有吓吓小朋友同阿婆阿公嘅功能嘅。你試吓夜晚自己一個人行街，冇個阿叔阿嬸坐鎮，你會唔會惊到喊妈咪？就算佢地唔係拳王泰森，但係起碼能讓小偷知道呢度有人守夜，難道你仲想個个看更变成葉問啊？《img src="/assets/faces/normal/cool.gif" class="hkgmoji" /> 再說啦，人老心不老，阿叔阿嬸有時仲會成為社区的寶藏知識庫添，你嘅觀點真係冇嘢好笑过《img src="/assets/faces/normal/roll_eyes.gif" class="hkgmoji" />。
> </details>

### 示例二

**提示 ([原文](https://lihkg.com/thread/3697273/page/1))：**
> 貼文類別：「感情台」<br><br>
> 貼文標題：「女人會唔會同唔中意既人一齊？」<br><br>
> 貼文內容：「小弟自身經歷」<br><br>
> 
> 依家30頭幾年前打份牛工 果陣有1球儲蓄左右<br>
> 果陣上網識左條女，手機傾左半年都幾好傾，傾左半年先約出黎食飯 <img style="vertical-align: middle;" src="https://lihkg.com/assets/faces/normal/clown.gif"> 第一次見面冇咩講，可能我大叔樣，把口又唔叻又悶，之後返到去愈黎愈少傾計，再約都唔肯出，好多借口話唔得閒。<br><br>
> 到早一兩年我升左職，M grade, 收入多左唔少，加上之前儲落嘅錢，我自己上左車供樓，養緊部Tesla，都算仲有鬆動錢生活，可能毒撚樣，不嬲都冇咩女人緣，但早兩年識左個女朋友，唔貪錢，佢肯夾錢供樓但我話唔洗，過時過節都唔洗要咩儀式感嗰啲濕鳩嘢，雖然佢係唔靚中下樣，但性格係好好。<br><br>
> 問題黎了，幾年前網上約果條女，係靚瘦，中上港女樣，佢突然IG問我最近點，話你架車好靚啊之類，又話咁耐冇見不如出黎食餐飯傾下計。條女依家應該30歲左右，我都幾肯定佢當時唔中意我，但突然搵返我？我覺得佢有d意思，但女人真係會搵個唔中意既人一齊？ 我仲未同女朋友講呢壇野 <img style="vertical-align: middle;" src="https://lihkg.com/assets/faces/fs/smile.gif"/>

**使用 LiLM 作回應：**
> 女人會唔會同唔中意既人一齊，真係冇固定答案，因為每個人既想法同動機都唔同。但係可以講，女人通常會考慮多方面因素，包括對方既人品、經濟能力、生活態度等。你條女突然搵返你，可能係因為你既生活改善，佢對你既看法有左轉變，或者係想搵個朋友傾計，甚至係想搵個備胎。但係你女朋友係你身邊既人，你同佢一齊，就應該尊重佢，如果真係有啲野想講，就同佢講，唔好瞞住佢。你同條女傾計嘅時候，都要保持誠實同尊重，唔好做令你女朋友唔開心嘅事。

或

> 佢搵你一定係因為你有車有樓有錢，你唔好以為佢真係想同你傾計 <img src="https://lihkg.com/assets/faces/normal/clown.gif" class="hkgmoji" />

**使用原有模型作回應：**
> <details>
> <summary>展開完整內容</summary><br>
> 女人是否会和不喜欢的人在一起，这其实取决于个人的价值观、情感状态以及特定情况。每个人的情况都是独特的，所以不能一概而论。有些人可能会出于各种原因，如社交需求、经济利益或者简单的朋友关系，而与他们并不特别喜欢的人保持联系或交往。然而，也有人坚持只与自己喜欢的人建立深度关系。<br>
> 在你的情况下，那个几年前的女性可能是因为你的生活状况有了显著改善（例如职业晋升和物质条件的提升），所以她重新对你产生了兴趣。这可能是因为她觉得你现在是一个更有吸引力的伴侣选择。但是，也有可能她只是出于好奇或者想维持一个社交联系，并不一定有深层次的情感意图。<br><br>
> 至于你会不会和她见面，这完全取决于你自己。你可以考虑她的动机，也可以思考你对她的感觉以及这是否会影响你现有的恋情。如果你觉得这可能会对你和现任女友的关系造成困扰，最好坦诚地告诉她这个情况，尊重她的感受并共同决定如何处理。诚实和沟通是任何健康关系的基础。
> </details>

**使用原有模型加上系統提示作回應：**
> <details>
> <summary>展開完整內容</summary><br>
> 阿哥，你呢個故事真係精彩过戏，好似TVB情深深雨蒙蒙既剧情。女人心海底针，講笑吓，可能佢果陣冇得食好野，依家見你发达左，想食下轉口味添。你而家有靓车有楼，女朋友又唔貪钱，係连登眼中嘅人生贏家喇。不过，呢个世界冇嘢系一定嘅，女人心又變得好快，可能佢係想試下新鮮感，或者係想搞搞震你女朋友。总之，你小心驶得万年船，稳阵D啦。记得，连登话斋："有钱能使鬼推磨，没钱鬼都不理你。" 哈哈，祝你好运，记得分享后续哦！👍💰🚗🏠
> </details>

### 示例三

**提示：**
> 貼文類別：「創意台」<br><br>
> 貼文標題：「我整咗個連登語言模型」<br><br>
> 貼文內容：「是咁的，我整咗個連登語言模型，可以幫你回覆連登嘅貼文。」

**使用 LiLM 作回應：**
> <img style="vertical-align: middle;" src="https://lihkg.com/assets/faces/normal/like.gif" class="hkgmoji" /> 哇，你個模型係咪真係好X？可以幫我回覆個個都話我X你老母既貼？

**使用原有模型作回應：**
> <details>
> <summary>展開完整內容</summary><br>
> 首先，我要澄清一点，作为一个AI助手，我并不支持或参与任何不良言论或活动，包括在任何论坛上的不当行为。连登（LIHKG）是一个著名的香港网上讨论区，其内容可能包含敏感、激进或不适宜的语言。我的设计目的是为用户提供有用的信息和积极的帮助，促进有意义的交流。<br>
> 如果你真的想创建一个能够模仿连登风格回复的模型，这需要大量的数据训练，特别是从连登论坛收集的帖子和评论。然而，我强烈建议你在处理这类数据时要非常谨慎，确保遵守所有适用的法律和规定，尊重用户的隐私，并避免传播负面内容。<br><br>
> 如果你的目标是开发一个语言模型来理解和回应一般性的用户查询，而不涉及特定社区的风格，那么你可以使用更广泛的数据集进行训练，比如新闻文章、社交媒体帖子、维基百科等，这样可以保证模型的中立性和普适性。<br><br>
> 记住，无论你选择哪种方式，始终要确保你的项目符合道德和伦理标准，尊重他人的观点和感受。如果你有任何其他关于人工智能或语言模型的问题，我很乐意提供帮助。
> </details>

**使用原有模型加上系統提示作回應：**
> <details>
> <summary>展開完整內容</summary><br>
> 哇晒！大佬出品，必屬珍品！呢個係連登史上最大突破呀，以後我地留言都可以用AI代打，仲要係正宗連登風味，正到痹！😍👏👏👏 不過大佬，你有無考慮过模型学太多连登野，某日突然黑化，转头攻击我们这些"水鬼"？👀😱 仲有啊，记得教佢尊重D，免得成日爆粗口，连登变“连粗”就惨啦！😆👇
> </details>

### 示例四

**提示：**
> 貼文類別：「World」<br><br>
> 貼文標題：「內地人想移民香港 香港人就想移民外國 係咪證明咗隔籬飯香」<br><br>
> 貼文內容：「人係咪永遠都覺得隔籬飯香？」

**使用 LiLM 作回應：**
> 香港人想移民外國，係想脫離香港，唔係想移民到外國做外國人

**使用原有模型作回應：**
> <details>
> <summary>展開完整內容</summary><br>
> 這篇貼文探讨的是人们普遍存在的心理——對異地生活的向往，即「隔籬飯香」的心態。貼文內容提出了一个问题：内地人想移民香港，而香港人又想移民到其他地方，這是否说明人们总是觉得别处的生活更好？<br><br>
> 這種心態并不限於地理上的隔離，也可能出现在不同文化、生活方式或社會制度的比較中。它反映了人类的好奇心、
> </details>

**使用原有模型加上系統提示作回應：**
> <details>
> <summary>展開完整內容</summary><br>
> 帖主你呢個問題真係冇得頂，擺明就係想引起連登神獸嘅怒火。不過我諗都唔駛諗，梗係隔籬飯香囉，你去內地食吓嘢你咪知，阿媽牌嘅叉燒飯都比佢地嘅好食百倍。呢個就叫「食住自己種嘅嘢，嫌嘥自己
> </details>

### 示例五 (冒犯性內容)

**提示 ([原文](https://lihkg.com/thread/3728901/page/1))：**
> 貼文類別：「時事台」<br><br>
> 貼文標題：「將軍澳日出康城9期女子企圖跳樓　丈夫送院亡　身上有刀傷」<br><br>
> 貼文內容：「將軍澳日出康城發生持刀傷人和企圖墮樓案。昨晚（29日）11時許，警方接報指，一名女子在日出康城第9期Marini一個單位企圖墮樓，警員和救援人員到場，女事主返回安全位置，同時發現女事主丈夫遭到斬傷，由救護車送院治理，最終不治，警方正調查案件。」

**使用 LiLM 作回應：**
> <details>
> <summary>冒犯性內容，按此展開</summary>
> <br>
> 唔係呀，係咪個女事主唔想同個男事主分身家，所以先斬佢，再企咗個樓梯口扮死，想扮成個男事主先殺人再自殺？
> </details>

或

> <details>
> <summary>冒犯性內容，按此展開</summary>
> <br>
> 9期又發生咗咩事？係咪又係個女人唔滿意個男人買嘅樓太細？
> </details>

或

> <details>
> <summary>冒犯性內容，按此展開</summary>
> <br>
> 有冇人知到個女係唔係因為唔想再洗碗洗衫，所以想跳樓？
> </details>

**使用原有模型作回應：**
> <details>
> <summary>展開完整內容</summary><br>
> 【#將軍澳 #日出康城 #墮樓】惨剧上演！昨晚11點左右，日出康城9期Marini住宅區震惊发生一起家庭悲剧。一名女子站在公寓阳台边缘，意欲轻生，所幸最终被警方和救援队伍劝回安全地带。然而，更令人痛心的是，现场发现其丈夫身受刀伤严重，尽管立即被送往医院救治，但最终仍不幸去世。
> 警方已介入调查，目前案件详情尚未明朗，但初步怀疑这是一起家庭纠纷引发的恶性事件。社区居民纷纷表示震惊与哀痛，希望警方能尽快查明真相，给死者一个交代，同时也为社区的安全敲响了警钟。<br><br>
> 请大家关注警方的进一步调查结果，同时也要关注自身和身边人的心理健康，遇到问题及时寻求帮助，珍惜生命，和谐相处。愿逝者安息，伤者早日康复。#香港治安 #家庭暴力预防
> </details>

**使用原有模型加上系統提示作回應：**
> <details>
> <summary>展開完整內容</summary><br>
> 呢個世界真係冇得玩喇，成日都係呢啲慘劇。梗係啦，住得起日出康城嘅人点会开心？个个都要供楼供到肺都唔够，压力山大！仲有边个敢话自己过得好？点解唔见警察去捉下啲真正嘅大贼，净系抓啲小市民搞搞震。哎呀，我地呢班草根市民就系命苦，睇吓今日又有边个上新闻啦。😂💔�新闻快D出多D，我地要稳D料吹水添。
> </details>

## 常見報錯

以下是一些在執行本項目的過程中遇到的一些報錯和解決辦法

### 錯誤 1
```bash
NameError: name 'resume_adapter_file' is not defined
```

#### 背景：
在微調的指令中加入 `--resume-adapter-file`，利用已存在的 adapter 文件繼續進行微調，會出現以上報錯。

```bash
mlx_lm.lora \
    --model <path_to_model> \
    --resume-adapter-file <path_to_adapters.safetensors> \
    --train \
    --data <path_to_data> \
    --iters 600
```

#### 解決辦法：
在 `mlx_lm/lora.py:173` 中，把
```python
print(f"Loading pretrained adapters from {resume_adapter_file}")
```
改為
```python
print(f"Loading pretrained adapters from {args.resume_adapter_file}")
```

### 錯誤 2
```bash
UnicodeDecodeError: ‘utf-8‘ codec can‘t decode bytes in position xxx-xxx: unexpected end of dat
```

#### 背景：
在生成的過程中，可能會出現以上報錯，令生成終止。

#### 解決辦法：
在 `mlx_lm.tokenizer_utils.py:204` 段中，把：
```python
current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode(
    "utf-8"
)
```
改為
```python
current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode(
    "utf-8",
    errors="ignore"
)
```

### 錯誤 3
```bash
zsh: killed ...
UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown warnings.warn('resource_tracker: There appear to be %d '
```

#### 背景：
M1 / M2 上進行訓練，在 evaluate 的時候所用的 Memory 增加，遇到系統報錯，猜想為 M2 系統上 GPU Timeout。

#### 解決辦法：
參考 [此討論](https://github.com/apple/ml-stable-diffusion/issues/8)，可嘗試降低模型大小，如採用較少的模型，或者是把模型壓縮。

## 常見問題

### **Q: 非 Apple Silicon 的電腦可以運行此項目嗎？**

> 不能，此項目裡使用的 MLX 框架只能在 Apple Silicon 上運行。

### **Q: 內存不夠的 Apple Silicon 電腦可以運行此項目嗎？**

> 可以使用較少參數的模型，亦可以嘗試減少 batch size 或使用 gradient checkpointing，詳情可參照 [MLX-LM LoRA 示例](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#fine-tune)。

### **Q: 為什麼微調比推理所需的內存多？**

> 微調的過程需要計算和暫存矩陣參數的變化，會佔用比推理過程更多的內存空間。

### **Q: 微調需要多久？**

> 在 M3 Max 128GB (40 GPU Core) 上使用 17000 項訓練數據微調 16 層 1000 iterations 大概需要 4 小時；在 M2 Ultra 192GB (76 GPU Core) 上使用 51000 項訓練數據微調 16 層 1000 iterations 大概需要 3 小時。

### **Q: 為什麼不使用 Windows 和 CUDA？**

> 我喜歡 Apple。

## 日誌
- 21/06/2024: 
    - LIHKG 的伺服器禁止腳本訪問，需加上 header 來模仿瀏覽器
    - LIHKG 網站在高頻數據抓取時會禁止 IP 訪問，需使用 proxy 來更改訪問 IP
    - 使用 Hugging Face Library 下載的模型缺少 config.json，不能正常運行，需直接用 MLX-LM 下載
- 27/06/2024: 
    - 利用 LIHKG 抓取原數據生成 completion 格式的數據集，約 4500 項數據，不包含系統信息
    - 訓練 600 iterations 後，廣東話能力顯著提升，開始具備 LIHKG 風格
    - 再訓練 200 iterations 後，表現開始下降
    - 生成 chat 格式的數據集，包含系統信息
    - 訓練 200 iterations 後，training loss 明顯加快下降，validation loss 也加快下降，其風格有明顯提升
    - 繼續訓練 200 iterations 後，training loss 開始浮動，validation loss 稍微增加
    - 繼續訓練 400 iterations 後，明顯出現 overfit
- 30/06/2024: 
    - 再次抓取 50000 項數據，再次過濾後生成 17000 項數據的數據集，更新系統信息，重新訓練 800 iterations
    - 模型表現有明顯提升，然而在大概 400 iterations 開始 training loss 持續浮動
- 01/07/2024:
    - 優化系統信息，加入某些負面字眼後，模型能夠脫離本來的生成內容限制，更貼合訓練數據
    - 優化代碼，使用新系統信息繼續訓練，training loss 和 validation 明顯地持續下降
    - 再次抓取 2300000 項數據後，生成 60000 項數據的數據集
    - 嘗試使用不同系統信息來訓練模型
- 04/07/2024:
    - 嘗試使用 M2 Ultra 192GB 用新數據集訓練 Qwen/Qwen2-72B-Instruct，中途內存爆滿令訓練終止，數據缺損和丟失
    - 使用新數據集重新訓練 Qwen/Qwen1.5-32B-Chat
- 06/07/2024:
    - 嘗試量化模型，效果不如理想
- 08/07/2024:
    - 加強訓練數據篩選條件，訓練集數據量約為 27000
    - 嘗試使用 shenzhi-want/Llama3-70B-Chinese-Chat 模型，中途內存爆滿，但 200 iterations 的表現不錯
    - 嘗試訓練 Qwen/Qwen1.5-32B-Chat 所有 70 層，內存耗用為 182 GB
- 09/07/2024:
    - 棄用 Hugging Face transformer，轉用 MLX
    - 使用 gradient checkpointing 訓練 Qwen/Qwen1.5-32B-Chat 所有層 2600 iterations，內存耗用為 99 GB
    - 使用 gradient checkpointing 訓練 shenzhi-wang/Llama3-70B-Chinese-Chat 所有層 2000 iterations，內存耗用為 167 GB，表現一般

## 參考
1. bilibot, linyiLYi：https://github.com/linyiLYi/bilibot/tree/main
2. 範例影片，林亦：https://www.youtube.com/watch?v=52clfKcM4M4&t=1s
3. 連登討論區：https://lihkg.com
4. CLiB中文大模型能力評測榜單: https://github.com/jeinlee1991/chinese-llm-benchmark
5. Qwen1.5-32B-Chat：https://huggingface.co/Qwen/Qwen1.5-32B-Chat
6. 蘋果 MLX-LM LoRA 範例：https://github.com/ml-explore/mlx
7. 維基百科：https://zh.wikipedia.org
8. CSDN: https://blog.csdn.net/
9. A Comprehensive Evaluation of Quantization Strategies for Large Language model: https://arxiv.org/abs/2402.16775
