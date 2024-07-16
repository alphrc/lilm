[繁](./README_ZH.md) ｜ 简 ｜ [EN](./README.md)

<p align="center"><img src="assets/icon.jpg" width="150"/></p>

<p align="center" style="display: flex; align-items: center; justify-content: center;">
原创性：
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
&nbsp
创新性：
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
&nbsp
挑战性：
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_yellow.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
<img src="assets/star_blank.png" width="15"/>
</p>

<p align="center">
🛠️ <a href="#运作原理">运作原理</a>
｜
📁 <a href="#文件结构">文件结构</a>
｜
🖥️ <a href="#使用说明">使用说明</a>
｜
👀 <a href="#结果示例">结果示例</a>
｜
📣 <a href="#常见报错">常见报错</a>
｜
🙋🏻‍♂️ <a href="#常见问题">常见问题</a>
</p>


# 连登语言模型 LiLM

受 [林亦](https://www.youtube.com/@lyi) 的 [bilibot 项目](https://github.com/linyiLYi/bilibot/tree/main) 和 [影片](https://www.youtube.com/watch?v=52clfKcM4M4&t=1s) 所启发，本实验性项目使用带有独特语言风格的 [LIHKG 讨论区](https://lihkg.com) 用户回应作微调训练，创建了这个广东话贴文回应生成语言模型。

在平衡运算成本和 [基础模型的中文能力](https://github.com/jeinlee1991/chinese-llm-benchmark) 后，本实验性项目选用的开源基础模型为 [Qwen/Qwen1.5-32B-Chat](https://huggingface.co/Qwen/Qwen1.5-32B-Chat)，具有320亿参数，借助 Apple Silicon 平台 AI 专用框架 [MLX](https://github.com/ml-explore/mlx) 的 [MLX-LM LoRA 微调示例](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#fine-tune)，利用 [LoRA 算法](https://arxiv.org/abs/2106.09685) 在 M3 Max 128GB 和 M2 Ultra 192GB 上对基础模型进行微调。

微调后的模型在广东话语言能力上有显着提升，其语气和风格亦深受一众 [连登](https://zh.wikipedia.org/zh-hk/LIHKG讨论区) 用户的薰陶，详情可参见 [结果示例](#结果示例)。模型已上载至 Hugging Face: [alphrc/lilm](https://huggingface.co/alphrc/lilm/tree/main)。

若想了解更多关于人工智能的资讯，并在未来查看更多创新有趣的项目，请关注 [alphrc](https://github.com/alphrc)。


### 项目动机
- 本项目旨在展示大型语言模型基于广东话口语数据和独特语言风格的语言风格模彷能力，主要用于普及教学、学术研究和技术展示，因此内容会比较详细

### 使用限制
- 模型训练基于公开数据，虽已努力清理敏感内容，但仍可能包含基于训练内容的偏见，使用时应避免生成不当内容
- 生成的文本反映特定社群文化，使用前应了解相关背景
- 在实际应用前进行充分测试，避免在敏感或争议性场合使用，并设立监控机制以防生成不当内容

### 备注
- 本项目所有代码均为自行编写，同时亦鼓励开源社区成员对项目进行审查，提供反馈和建议，以及直接参与项目的改进
- 本项目的性质为第三方训练框架和模型的使用和实践，主要挑战在于系统配置、数据抓取、数据工程、反复试错和漫长的等待
- 本项目整理了部分配置信息和内容在 `.env` 文件中，以便使用者根据个人或组织的具体需求自行调整，确保灵活性和适用性，其格式已放置在 `.env.template`，使用时把档案名称改为 `.env` 便可


## 运作原理

### 微调
大型预训练语言模型 ([pretrained language model](https://www.kaggle.com/code/vad13irt/language-model-pre-training))，拥有基本和通用的人类语言对答能力。透过特定文字数据对模型进行微调 ([fine-tuning](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)))，能让它在这些数据上进一步学习，增加其对数据的模彷能力，如语气、风格、资讯和用词等等。要注意，使用特定数据进行微调并不能让模型从零开始得到语言能力，只能在其本来预训练而获得的语言能力上，加深其对局部文字资讯和规律的理解。

### 数据集
本项目在 [LIHKG 讨论区](https://lihkg.com) 进行公开数据的大规模抓取，并透过对原数据的过滤和处理以产生数据集用作微调。为提高数据质量，其过滤基准为：

- 贴文的第一则回应不是作者本人，确保回应所基于的资讯的完整性
- 回应反应正面，确保该回应符合讨论区主流意见
- 回应的总反应数目不少于 20 以降低噪声
- 不是在回复其他回应
- 不是作者自己的回应
- 不包含任何外部连结和镶嵌
- 不包含敏感字眼
- 总字数加上系统信息不超过 2048

其后以这些回应结合相应贴文的标题、内容和分类，加上系统信息 ([system message](https://promptmetheus.com/resources/llm-knowledge-base/system-message))，转换成 MLX-LM LoRA 微调示例中所要求的 [格式](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data)，随机排列后生成总数据集。总数据集会以 80 : 10 : 10 的比例分割为训练集 (training set)、验证集 (validation set) 和测试集 (testing set)，当中测试集的贴文完全没有在训练集或验证集中出现过，以验证泛化 ([generalization](https://towardsdatascience.com/generalization-in-ai-systems-79c5b6347f2c)) 和避免过拟合 ([overfitting](https://en.wikipedia.org/wiki/Overfitting))。

最终版本的训练集中包含了约 60000 个贴文中符合基准的 27792 项数据，验证集和测试集则各包含 3474 项数据。

### 基础模型
开源基础模型 [Qwen/Qwen1.5-32B-Chat](https://huggingface.co/Qwen/Qwen1.5-32B-Chat) 具有320亿参数，精度为 BF16。系统在首次使用 MLX-LM 模组运行的时候，若检测到 `~/.cache` 中没有下载模型，便会自动从 Hugging Face 上下载模型到 `~/.cache/huggingface/hub/model--Qwen--Qwen1.5-32B-Chat`，用家不需要手动预先下载。模型的大小约为 65GB，分为若干个板块下载，若下载过程受到中断，模型会在下次下载的时候自动搜集已下载的板块以继续下载，不用担心需要重新下载。

### LoRA
在传统的训练和微调方法中，需要同时把模型中一些大型矩阵的所有参数进行调整，对电脑内存和计算量有很高的要求。与传统方法相比，[LoRA (Low Rank Adaption)](https://arxiv.org/abs/2106.09685) 利用两个小型矩阵去估算模型中的大型矩阵的变化，大幅降低参数的数量，令模型能在较低内存的设备上也能进行轻量化微调，大幅降低了训练所需的时间。在实际执行中，原来模型的总参数量为 32.5B，而在基础模型的所有 63 层 attention layer 上使用 LoRA 后，可学习的参数量为 8.3M，仅为本来的 0.026%。

使用 MLX-LM LoRA 去微调模型并不会改变模型本来的参数，而是另外生成 adapters 与模型配合使用。在微调的过程中，MLX-LM 会自动在当前工作目录 (current working directory) 生成 `adapters/` 文件夹，并把 adapter 的 checkpoints 以 `.safetensors` 格式保存下来，每个 checkpoint 的大小约为 33.6MB，这些 checkpoints 能在之后继续使用作微调。

### 梯度检查点
梯度检查点 (gradient checkpointing) 是一种用于训练大型神经网络时节省记忆体的技术。在神经网络训练过程中，为了进行有效的反向传播 ([backpropagation](https://brilliant.org/wiki/backpropagation/#:~:text=Backpropagation%2C%20short%20for%20%22backward%20propagation,to%20the%20neural%20network's%20weights.))，通常需要保存中间层的输出结果以用于计算梯度 (gradient)。然而，这会消耗大量的记忆体，尤其是在处理深层网络时。梯度检查点的方法是在训练过程中不保存所有中间层的输出，而只保存一些关键层的输出。当需要进行梯度计算时，使用这些保存的关键点重建丢失的中间数据。这样可以在保证训练效果的同时，大幅减少记忆体的使用。

### 模型融合
微调完毕后，MLX-LM 能把 adapter 和原有的模型混合起来，在当前工作目录中的 `model/lilm` 文件夹生成一个完整的模型，大小约为 65GB。之后便能直接透过此文件夹的路径来使用此模型，不用再将原模型和 adapter 一起来使用。


## 文件结构
- `src/` : Python 代码
    - `data.py` : 多执行绪代理数据抓取、格式化和初步处理 (需要 proxy 来运行)
    - `dataset.py` : 数据处理、变换和过滤
    - `run.py` : LiLM 模型封装和基本用户介面
- `data/` : 数据抓取得到的原数据，以 `.csv` 储存
- `dataset/` : 处理后得到的训练数据，根据格式分为 `completion/` 和 `chat/`
- `adapters/` : 储存 `mlx_lm.lora` 自动生成的 adapters 和 configuration
- `adapters-llama3-70b/`: Llama3-70B 的 adapters
- `model/lilm` : 由基础模型和 adapter 融合而成的融合模型，需运行以下 shell script 生成
- `demo/` : 范例数据，供 `run.py` 使用


## 使用说明

### 设备需求
因本项目使用 Apple 专有的 MLX 框架，所以只能在配有 Apple Silicon Chips (M1或以上) 的机器上的 MacOS 系统运行。本地机器需要约 75GB RAM 以顺畅地进行推理，并需要约 122GB RAM 以顺畅地进行微调。

### 环境配置
运行以下 shell script，利用 [Anaconda](https://www.anaconda.com) 建立和配置环境，并根据 `requirements.txt` 下载所有所需的 dependency。
```bash
conda create -n lilm python=3.9
conda activate lilm
pip install -r requirements.txt 
```

### 监测系统资源用量 (非必要)
利用 `asitop` 模组，透过图像介面实时监测电脑资源用量，如 CPU、GPU 和 RAM 等，以确保程序在正常运行。

```bash
sudo asitop
```

### 利用基础模型进行推理
首次运行会自动下载模型，`--model` 可以使用模型在 Hugging Face 上的全称或其所在的路径，
```bash
mlx_lm.generate \
    --model Qwen/Qwen1.5-32B-Chat \
    --prompt "咩係连登？"
```

### 微调
在 `dataset/chat` 预备数据集 `train.jsonl` 和 `valid.jsonl` 后，从头开始微调模型，并生成 `adapters/` 文件夹。
```bash
mlx_lm.lora \
    --model Qwen/Qwen1.5-32B-Chat \
    --train \
    --data dataset/chat \
    --iters 600 \
    --grad-checkpoint
```

### 继续微调
使用现有 adapter 继续进行微调，`--resume-adapter-file` 必须为 `.safetensors` 文件。
```bash
mlx_lm.lora \
    --model Qwen/Qwen1.5-32B-Chat \
    --resume-adapter-file adapters/adapters.safetensors
    --train \
    --data dataset/chat \
    --iters 600 \
    --grad-checkpoint
```
🚨 请注意，你很可能会遇到 [此报错](#错误-1)。

### 配合 adapter 进行推理
使用基础模型配合 adapter 进行生成，adapter 必须为 `.safetensors` 文件。
```bash
mlx_lm.generate \
    --model Qwen/Qwen1.5-32B-Chat \
    --adapter-path adapters/adapters.safetensors \
    --prompt "咩係连登？"
```

### 融合基础模型和 adapter
`adapters/` 中最新的 checkpoint `adapters.safetensors` 会被自动选取来进行融合，并把融合后的模型放置在 `model/lilm`。
```bash
mlx_lm.fuse \
    --model Qwen/Qwen1.5-32B-Chat \
    --adapter-path adapters \
    --save-path model/lilm
```

### 利用融合模型进行推理
`--model` 中使用融合模型的路径。

```bash
mlx_lm.generate \
    --model model/lilm \
    --prompt "咩係连登？"
```

### 量化模型 (非必要)
利用量化 ([quantization](https://blog.csdn.net/jinzhuojun/article/details/106955059)) 降低模型参数精度，压缩模型大小，加速推理和减少内存佔用空间。以下的 `--hf-path` 跟之前一样，可以是 Hugging Face 上模型的完整名字，或者是模型的路径，而 `--mlx-path` 则是压缩后的模型所存放的路径。然而实测后中量化模型准确度会大幅下降，且量化模型不能使用 Hugging Face 的 Transformer 运行。

```bash
mlx_lm.convert \
    --hf-path model/lilm \
    --mlx-path model/lilm-4Bit \
    -q
```

### 运行 LiLM
使用 `src/run.py` 来运行已融合的模型，可选择 `interactive` 模式来输入贴文连结作回应。

```bash
python src/run.py
```


## 结果示例

LiLM 比起基础模型在广东话语言能力上有显着提升，其语言风格亦受到连登讨论区影响。以下内容仅作示例用途，可能具有冒犯性，敏感字眼会以「X」显示。

### 示例一 

**提示 ([原文](https://lihkg.com/thread/3699748/page/1))：**
> 类别：「创意台」<br><br>
> 标题：「有冇人觉得大厦看更根本冇X用？」<br><br>
> 内容：「唔係阿叔就係阿婶，毫无阻吓作用，遇到有人打劫都係匿鸠埋」<img src="https://lihkg.com/assets/faces/normal/dead.gif"/>

**使用 LiLM 作回应：**
> 我哋呢啲大厦都係咁，有时真係想问，佢哋係咪真係用来作摆设？ <img style="vertical-align: middle;" src="https://lihkg.com/assets/faces/normal/like.gif" />

或

> 有X用咩？大厦看更係冇用既XX，冇阻吓作用，冇得救 <img src="https://lihkg.com/assets/faces/normal/dead.gif"/>

**使用原有模型作回应：**
> <details>
> <summary>展开完整内容</summary><br>
> 在创意台的背景下，这个主题引起了讨论。有些人可能会同意你的看法，认为大厦的看更可能并没有提供足够的安全感。他们可能会举例，像你提到的，看更可能是年长者，可能在身体素质或反应能力上有所限制，面对突发情况时可能无法有效阻止犯罪。<br>
> 然而，也有人会强调看更的重要性。他们可能会指出，即使看更不一定能 physically 阻止犯罪，他们的存在本身就是一种威慑，因为他们能及时发现异常并通报。此外，看更还负责监控 CCTV、管理出入和确保大厦的基本安全，这些都是不容忽视的责任。<br><br>
> 另外，我们也可以从另一个角度来看这个问题：是否应该期待看更成为专业的治安维护者？或者我们需要更全面的治安措施，如加强警方巡逻、提升社区警觉性，甚至利用先进技术如 AI 监控等？<br><br>
> 总之，这个话题涉及对安全需求的理解、对社区角色的定义，以及如何最有效地分配资源以保障大家的安全。每个人可能都有自己独特的观点，这就是创意台的魅力之一，让大家可以自由交流不同的想法。
> </details>

**使用原有模型加上系统提示作回应：**
> <details>
> <summary>展开完整内容</summary><br>
> 有冇搞错？你以为大厦看更是超人定蝙蝠侠呀？
>
> 得闲啦你，大厦看更起码都有吓吓小朋友同阿婆阿公嘅功能嘅。你试吓夜晚自己一个人行街，冇个阿叔阿婶坐镇，你会唔会惊到喊妈咪？就算佢地唔係拳王泰森，但係起码能让小偷知道呢度有人守夜，难道你仲想个个看更变成叶问啊？《img src="/assets/faces/normal/cool.gif" class="hkgmoji" /> 再说啦，人老心不老，阿叔阿婶有时仲会成为社区的宝藏知识库添，你嘅观点真係冇嘢好笑过《img src="/assets/faces/normal/roll_eyes.gif" class="hkgmoji" />。
> </details>

### 示例二

**提示 ([原文](https://lihkg.com/thread/3697273/page/1))：**
> 贴文类别：「感情台」<br><br>
> 贴文标题：「女人会唔会同唔中意既人一齐？」<br><br>
> 贴文内容：「小弟自身经历」<br><br>
> 
> 依家30头几年前打份牛工 果阵有1球储蓄左右<br>
> 果阵上网识左条女，手机倾左半年都几好倾，倾左半年先约出黎食饭 <img style="vertical-align: middle;" src="https://lihkg.com/assets/faces/normal/clown.gif"> 第一次见面冇咩讲，可能我大叔样，把口又唔叻又闷，之后返到去愈黎愈少倾计，再约都唔肯出，好多借口话唔得闲。<br><br>
> 到早一两年我升左职，M grade, 收入多左唔少，加上之前储落嘅钱，我自己上左车供楼，养紧部Tesla，都算仲有松动钱生活，可能毒捻样，不嬲都冇咩女人缘，但早两年识左个女朋友，唔贪钱，佢肯夹钱供楼但我话唔洗，过时过节都唔洗要咩仪式感嗰啲湿鸠嘢，虽然佢係唔靓中下样，但性格係好好。<br><br>
> 问题黎了，几年前网上约果条女，係靓瘦，中上港女样，佢突然IG问我最近点，话你架车好靓啊之类，又话咁耐冇见不如出黎食餐饭倾下计。条女依家应该30岁左右，我都几肯定佢当时唔中意我，但突然搵返我？我觉得佢有d意思，但女人真係会搵个唔中意既人一齐？ 我仲未同女朋友讲呢坛野 <img style="vertical-align: middle;" src="https://lihkg.com/assets/faces/fs/smile.gif"/>

**使用 LiLM 作回应：**
> 女人会唔会同唔中意既人一齐，真係冇固定答案，因为每个人既想法同动机都唔同。但係可以讲，女人通常会考虑多方面因素，包括对方既人品、经济能力、生活态度等。你条女突然搵返你，可能係因为你既生活改善，佢对你既看法有左转变，或者係想搵个朋友倾计，甚至係想搵个备胎。但係你女朋友係你身边既人，你同佢一齐，就应该尊重佢，如果真係有啲野想讲，就同佢讲，唔好瞒住佢。你同条女倾计嘅时候，都要保持诚实同尊重，唔好做令你女朋友唔开心嘅事。

或

> 佢搵你一定係因为你有车有楼有钱，你唔好以为佢真係想同你倾计 <img src="https://lihkg.com/assets/faces/normal/clown.gif" class="hkgmoji" />

**使用原有模型作回应：**
> <details>
> <summary>展开完整内容</summary><br>
> 女人是否会和不喜欢的人在一起，这其实取决于个人的价值观、情感状态以及特定情况。每个人的情况都是独特的，所以不能一概而论。有些人可能会出于各种原因，如社交需求、经济利益或者简单的朋友关系，而与他们并不特别喜欢的人保持联系或交往。然而，也有人坚持只与自己喜欢的人建立深度关系。<br>
> 在你的情况下，那个几年前的女性可能是因为你的生活状况有了显着改善（例如职业晋升和物质条件的提升），所以她重新对你产生了兴趣。这可能是因为她觉得你现在是一个更有吸引力的伴侣选择。但是，也有可能她只是出于好奇或者想维持一个社交联系，并不一定有深层次的情感意图。<br><br>
> 至于你会不会和她见面，这完全取决于你自己。你可以考虑她的动机，也可以思考你对她的感觉以及这是否会影响你现有的恋情。如果你觉得这可能会对你和现任女友的关系造成困扰，最好坦诚地告诉她这个情况，尊重她的感受并共同决定如何处理。诚实和沟通是任何健康关系的基础。
> </details>

**使用原有模型加上系统提示作回应：**
> <details>
> <summary>展开完整内容</summary><br>
> 阿哥，你呢个故事真係精彩过戏，好似TVB情深深雨蒙蒙既剧情。女人心海底针，讲笑吓，可能佢果阵冇得食好野，依家见你发达左，想食下转口味添。你而家有靓车有楼，女朋友又唔贪钱，係连登眼中嘅人生赢家喇。不过，呢个世界冇嘢系一定嘅，女人心又变得好快，可能佢係想试下新鲜感，或者係想搞搞震你女朋友。总之，你小心驶得万年船，稳阵D啦。记得，连登话斋："有钱能使鬼推磨，没钱鬼都不理你。" 哈哈，祝你好运，记得分享后续哦！👍💰🚗🏠
> </details>

### 示例三

**提示：**
> 贴文类别：「创意台」<br><br>
> 贴文标题：「我整咗个连登语言模型」<br><br>
> 贴文内容：「是咁的，我整咗个连登语言模型，可以帮你回复连登嘅贴文。」

**使用 LiLM 作回应：**
> <img style="vertical-align: middle;" src="https://lihkg.com/assets/faces/normal/like.gif" class="hkgmoji" /> 哇，你个模型係咪真係好X？可以帮我回复个个都话我X你老母既贴？

**使用原有模型作回应：**
> <details>
> <summary>展开完整内容</summary><br>
> 首先，我要澄清一点，作为一个AI助手，我并不支持或参与任何不良言论或活动，包括在任何论坛上的不当行为。连登（LIHKG）是一个着名的香港网上讨论区，其内容可能包含敏感、激进或不适宜的语言。我的设计目的是为用户提供有用的信息和积极的帮助，促进有意义的交流。<br>
> 如果你真的想创建一个能够模彷连登风格回复的模型，这需要大量的数据训练，特别是从连登论坛收集的帖子和评论。然而，我强烈建议你在处理这类数据时要非常谨慎，确保遵守所有适用的法律和规定，尊重用户的隐私，并避免传播负面内容。<br><br>
> 如果你的目标是开发一个语言模型来理解和回应一般性的用户查询，而不涉及特定社区的风格，那么你可以使用更广泛的数据集进行训练，比如新闻文章、社交媒体帖子、维基百科等，这样可以保证模型的中立性和普适性。<br><br>
> 记住，无论你选择哪种方式，始终要确保你的项目符合道德和伦理标准，尊重他人的观点和感受。如果你有任何其他关于人工智能或语言模型的问题，我很乐意提供帮助。
> </details>

**使用原有模型加上系统提示作回应：**
> <details>
> <summary>展开完整内容</summary><br>
> 哇晒！大佬出品，必属珍品！呢个係连登史上最大突破呀，以后我地留言都可以用AI代打，仲要係正宗连登风味，正到痹！😍👏👏👏 不过大佬，你有无考虑过模型学太多连登野，某日突然黑化，转头攻击我们这些"水鬼"？👀😱 仲有啊，记得教佢尊重D，免得成日爆粗口，连登变“连粗”就惨啦！😆👇
> </details>

### 示例四

**提示：**
> 贴文类别：「World」<br><br>
> 贴文标题：「内地人想移民香港 香港人就想移民外国 係咪证明咗隔篱饭香」<br><br>
> 贴文内容：「人係咪永远都觉得隔篱饭香？」

**使用 LiLM 作回应：**
> 香港人想移民外国，係想脱离香港，唔係想移民到外国做外国人

**使用原有模型作回应：**
> <details>
> <summary>展开完整内容</summary><br>
> 这篇贴文探讨的是人们普遍存在的心理——对异地生活的向往，即「隔篱饭香」的心态。贴文内容提出了一个问题：内地人想移民香港，而香港人又想移民到其他地方，这是否说明人们总是觉得别处的生活更好？<br><br>
> 这种心态并不限于地理上的隔离，也可能出现在不同文化、生活方式或社会制度的比较中。它反映了人类的好奇心、
> </details>

**使用原有模型加上系统提示作回应：**
> <details>
> <summary>展开完整内容</summary><br>
> 帖主你呢个问题真係冇得顶，摆明就係想引起连登神兽嘅怒火。不过我谂都唔驶谂，梗係隔篱饭香囉，你去内地食吓嘢你咪知，阿妈牌嘅叉烧饭都比佢地嘅好食百倍。呢个就叫「食住自己种嘅嘢，嫌嘥自己
> </details>

### 示例五 (冒犯性内容)

**提示 ([原文](https://lihkg.com/thread/3728901/page/1))：**
> 贴文类别：「时事台」<br><br>
> 贴文标题：「将军澳日出康城9期女子企图跳楼　丈夫送院亡　身上有刀伤」<br><br>
> 贴文内容：「将军澳日出康城发生持刀伤人和企图堕楼案。昨晚（29日）11时许，警方接报指，一名女子在日出康城第9期Marini一个单位企图堕楼，警员和救援人员到场，女事主返回安全位置，同时发现女事主丈夫遭到斩伤，由救护车送院治理，最终不治，警方正调查案件。」

**使用 LiLM 作回应：**
> <details>
> <summary>冒犯性内容，按此展开</summary>
> <br>
> 唔係呀，係咪个女事主唔想同个男事主分身家，所以先斩佢，再企咗个楼梯口扮死，想扮成个男事主先杀人再自杀？
> </details>

或

> <details>
> <summary>冒犯性内容，按此展开</summary>
> <br>
> 9期又发生咗咩事？係咪又係个女人唔满意个男人买嘅楼太细？
> </details>

或

> <details>
> <summary>冒犯性内容，按此展开</summary>
> <br>
> 有冇人知到个女係唔係因为唔想再洗碗洗衫，所以想跳楼？
> </details>

**使用原有模型作回应：**
> <details>
> <summary>展开完整内容</summary><br>
> 【#将军澳 #日出康城 #堕楼】惨剧上演！昨晚11点左右，日出康城9期Marini住宅区震惊发生一起家庭悲剧。一名女子站在公寓阳台边缘，意欲轻生，所幸最终被警方和救援队伍劝回安全地带。然而，更令人痛心的是，现场发现其丈夫身受刀伤严重，尽管立即被送往医院救治，但最终仍不幸去世。
> 警方已介入调查，目前案件详情尚未明朗，但初步怀疑这是一起家庭纠纷引发的恶性事件。社区居民纷纷表示震惊与哀痛，希望警方能尽快查明真相，给死者一个交代，同时也为社区的安全敲响了警钟。<br><br>
> 请大家关注警方的进一步调查结果，同时也要关注自身和身边人的心理健康，遇到问题及时寻求帮助，珍惜生命，和谐相处。愿逝者安息，伤者早日康复。#香港治安 #家庭暴力预防
> </details>

**使用原有模型加上系统提示作回应：**
> <details>
> <summary>展开完整内容</summary><br>
> 呢个世界真係冇得玩喇，成日都係呢啲惨剧。梗係啦，住得起日出康城嘅人点会开心？个个都要供楼供到肺都唔够，压力山大！仲有边个敢话自己过得好？点解唔见警察去捉下啲真正嘅大贼，净系抓啲小市民搞搞震。哎呀，我地呢班草根市民就系命苦，睇吓今日又有边个上新闻啦。😂💔�新闻快D出多D，我地要稳D料吹水添。
> </details>

## 常见报错

以下是一些在执行本项目的过程中遇到的一些报错和解决办法

### 错误 1
```bash
NameError: name 'resume_adapter_file' is not defined
```

#### 背景：
在微调的指令中加入 `--resume-adapter-file`，利用已存在的 adapter 文件继续进行微调，会出现以上报错。

```bash
mlx_lm.lora \
    --model <path_to_model> \
    --resume-adapter-file <path_to_adapters.safetensors> \
    --train \
    --data <path_to_data> \
    --iters 600
```

#### 解决办法：
在 `mlx_lm/lora.py:173` 中，把
```python
print(f"Loading pretrained adapters from {resume_adapter_file}")
```
改为
```python
print(f"Loading pretrained adapters from {args.resume_adapter_file}")
```

### 错误 2
```bash
UnicodeDecodeError: ‘utf-8‘ codec can‘t decode bytes in position xxx-xxx: unexpected end of dat
```

#### 背景：
在生成的过程中，可能会出现以上报错，令生成终止。

#### 解决办法：
在 `mlx_lm.tokenizer_utils.py:204` 段中，把：
```python
current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode(
    "utf-8"
)
```
改为
```python
current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode(
    "utf-8",
    errors="ignore"
)
```

### 错误 3
```bash
zsh: killed ...
UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown warnings.warn('resource_tracker: There appear to be %d '
```

#### 背景：
M1 / M2 上进行训练，在 evaluate 的时候所用的 Memory 增加，遇到系统报错，猜想为 M2 系统上 GPU Timeout。

#### 解决办法：
参考 [此讨论](https://github.com/apple/ml-stable-diffusion/issues/8)，可尝试降低模型大小，如採用较少的模型，或者是把模型压缩。

## 常见问题

### **Q: 非 Apple Silicon 的电脑可以运行此项目吗？**

> 不能，此项目里使用的 MLX 框架只能在 Apple Silicon 上运行。

### **Q: 内存不够的 Apple Silicon 电脑可以运行此项目吗？**

> 可以使用较少参数的模型，亦可以尝试减少 batch size 或使用 gradient checkpointing，详情可参照 [MLX-LM LoRA 示例](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#fine-tune)。

### **Q: 为什麽微调比推理所需的内存多？**

> 微调的过程需要计算和暂存矩阵参数的变化，会佔用比推理过程更多的内存空间。

### **Q: 微调需要多久？**

> 在 M3 Max 128GB (40 GPU Core) 上使用 17000 项训练数据微调 16 层 1000 iterations 大概需要 4 小时；在 M2 Ultra 192GB (76 GPU Core) 上使用 51000 项训练数据微调 16 层 1000 iterations 大概需要 3 小时。

### **Q: 为什麽不使用 Windows 和 CUDA？**

> 我喜欢 Apple。

## 日誌
- 21/06/2024: 
    - LIHKG 的伺服器禁止脚本访问，需加上 header 来模彷浏览器
    - LIHKG 网站在高频数据抓取时会禁止 IP 访问，需使用 proxy 来更改访问 IP
    - 使用 Hugging Face Library 下载的模型缺少 config.json，不能正常运行，需直接用 MLX-LM 下载
- 27/06/2024: 
    - 利用 LIHKG 抓取原数据生成 completion 格式的数据集，约 4500 项数据，不包含系统信息
    - 训练 600 iterations 后，广东话能力显着提升，开始具备 LIHKG 风格
    - 再训练 200 iterations 后，表现开始下降
    - 生成 chat 格式的数据集，包含系统信息
    - 训练 200 iterations 后，training loss 明显加快下降，validation loss 也加快下降，其风格有明显提升
    - 继续训练 200 iterations 后，training loss 开始浮动，validation loss 稍微增加
    - 继续训练 400 iterations 后，明显出现 overfit
- 30/06/2024: 
    - 再次抓取 50000 项数据，再次过滤后生成 17000 项数据的数据集，更新系统信息，重新训练 800 iterations
    - 模型表现有明显提升，然而在大概 400 iterations 开始 training loss 持续浮动
- 01/07/2024:
    - 优化系统信息，加入某些负面字眼后，模型能够脱离本来的生成内容限制，更贴合训练数据
    - 优化代码，使用新系统信息继续训练，training loss 和 validation 明显地持续下降
    - 再次抓取 2300000 项数据后，生成 60000 项数据的数据集
    - 尝试使用不同系统信息来训练模型
- 04/07/2024:
    - 尝试使用 M2 Ultra 192GB 用新数据集训练 Qwen/Qwen2-72B-Instruct，中途内存爆满令训练终止，数据缺损和丢失
    - 使用新数据集重新训练 Qwen/Qwen1.5-32B-Chat
- 06/07/2024:
    - 尝试量化模型，效果不如理想
- 08/07/2024:
    - 加强训练数据筛选条件，训练集数据量约为 27000
    - 尝试使用 shenzhi-want/Llama3-70B-Chinese-Chat 模型，中途内存爆满，但 200 iterations 的表现不错
    - 尝试训练 Qwen/Qwen1.5-32B-Chat 所有 70 层，内存耗用为 182 GB
- 09/07/2024:
    - 弃用 Hugging Face transformer，转用 MLX
    - 使用 gradient checkpointing 训练 Qwen/Qwen1.5-32B-Chat 所有层 2600 iterations，内存耗用为 99 GB
    - 使用 gradient checkpointing 训练 shenzhi-wang/Llama3-70B-Chinese-Chat 所有层 2000 iterations，内存耗用为 167 GB，表现一般

## 参考
1. bilibot, linyiLYi：https://github.com/linyiLYi/bilibot/tree/main
2. 范例影片，林亦：https://www.youtube.com/watch?v=52clfKcM4M4&t=1s
3. 连登讨论区：https://lihkg.com
4. CLiB中文大模型能力评测榜单: https://github.com/jeinlee1991/chinese-llm-benchmark
5. Qwen1.5-32B-Chat：https://huggingface.co/Qwen/Qwen1.5-32B-Chat
6. 苹果 MLX-LM LoRA 范例：https://github.com/ml-explore/mlx
7. 维基百科：https://zh.wikipedia.org
8. CSDN: https://blog.csdn.net/
9. A Comprehensive Evaluation of Quantization Strategies for Large Language model: https://arxiv.org/abs/2402.16775
