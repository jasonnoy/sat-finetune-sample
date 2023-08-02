import torch
from sat import AutoModel
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy
from sat.model.official import ChatGLM2Model
from sat.model.finetune.lora2 import LoraMixin


class FineTuneModel(ChatGLM2Model):
    def __init__(self, args, transformer=None, parallel_output=True, **kw_args):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kw_args)
        self.add_mixin("lora", LoraMixin(args.num_layers, 10), reinit=True)

def chat(query, model, tokenizer, 
        max_length: int = 256, num_beams=1, top_p=0.7, top_k=0, temperature=0.95):
    prompt = f"[Round 0]\n\n问：{query}\n\n答："
    inputs = tokenizer([prompt], return_tensors="pt").to(model.parameters().__next__().device)['input_ids'][0]
    seq = torch.cat(
        [inputs, torch.tensor([-1]*(max_length-len(inputs)), device=inputs.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=0, end_tokens=[tokenizer.eos_token_id])
    strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=0, end_tokens=[tokenizer.eos_token_id], num_beams=num_beams, consider_end=True)
    
    output = filling_sequence(
        model, seq,
        batch_size=1,
        strategy=strategy
    )[0] # drop memory
    
    # ---------------
    # port from inference_glm.py, more general than chat mode
    # clip -1s and fill back generated things into seq
    output_list = list(output)

    response = tokenizer.decode(output_list[0])
    print(response)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    # load model
    # model, model_args = FineTuneModel.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690270701725/checkpoints/finetune-chatglm-6b-lora-07-25-15-39', 
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690281036622/checkpoints/finetune-chatglm2-6b-07-25-18-31',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690278447134/checkpoints/finetune-chatglm-6b-lora-07-25-17-48',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690281747909/checkpoints/finetune-chatglm2-6b-07-25-18-53',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690296760278/checkpoints/finetune-chatglm2-6b-07-25-22-53',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/jjh/projects/ftsample/checkpoints/finetune-chatglm2-6b-07-27-10-19',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/jjh/projects/ftsample/checkpoints/finetune-chatglm2-6b-07-27-15-05',
    model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/jjh/projects/ftsample/checkpoints/finetune-chatglm2-6b-08-02-11-02',
        args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True,
    ))
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    qlist=['日本lamp/蓝普隐形三维可调柜门铰链合页飞机烟斗橱柜', '唯美度假沙滩景色高清图片 素材中国16素材网', '严选现采大甲厚切炸芋头', '小罐茶全国首家现代茶生活体验馆落址西安', '激情促销 北京现代全新胜达最高可优惠8万现金 面向全国销售', '小香风气质连衣裙女2021夏季新款法式小黑裙宽松显瘦中长裙子', 'chali茶里 蜜桃乌龙铁观音茶包花果花草水果茶白桃乌龙茶叶冷泡', '文字文字广告和促销.业务概念的行动, 以刺激客户购买', '60粒百合康牌芦荟软胶囊 通便 澳能系列', '外汇交易的12堂必修课', '佳得乐助力2018年姚基金希望小学篮球季圆满成功', '图④ 专职维护旅游市场秩序和治安环境的三亚市旅游警察整装待发.', '商用点心蒸屉竹制小笼包 蒸饭笼 不锈钢包边蒸格广式早茶餐厅蒸笼', '【二手9成新】老年人体育健身指南 邓树勋 主编 广州体育科学学会', '腾讯奇迹觉醒翅膀怎么培养,升星?', '温野菜日式火锅生鸡蛋蘸料图片 第10533张', '收纳储物调料层整理架支架调味桌面物架架架架子2置架层角3厨房', '高血压专业诊治常规/中国医师协会高血压专业委员会', '亚克力工位牌姓名牌岗位牌 办公桌座位牌职位牌屏风挂牌员工牌挂式', '意蜂隔王板 养蜂工具 蜜蜂隔皇器 蜂箱木质隔王板 带包边隔王', 'totachi 道嘉驰 中国总代理', '1音箱 音响 电脑音箱 电视音响', '成品闲章【大雅】书画藏书法毛笔国画引首押尾手工篆刻定制作印章', '12宝宝的耐防水个月假玩婴幼儿可启蒙婴儿摔玩具咬', '高清组图:全运会摔跤男子古典式130公斤级决赛,聂晓明', '草莓爆爆珠水果味爆蛋爆浆珠果酱冰粉爆爆珠蛋珍珠奶茶店专用商用', '哑铃形试样裁切刀 橡胶哑铃裁刀', '淡橘色烘焙店招聘漫画风x展架', '火影忍者中鸣人是被谁五行封印的', '推拉门拉门衣柜门衣专业安装制作推拉门衣柜门安装每平方60 80元', '特大好消息!寒假将至,济南6大滑雪场对这些人滑雪免费', '芝麻白荔枝面', '虎头帽06', '爱的甜蜜_蛋糕西饼_莱西蛋糕,莱西生日蛋糕,青岛莱西', '侍魂2 川a大战407 霸王丸使出满屏超必杀天霸凄惶斩!', '男士内裤丝袜套阴茎透明丝袜油光亮丝打飞机套大小号可量身2018', '合肥消防业务技能大比拼', '45岁大叔种植大蒜亏损30万种植果树年入百万', 'yamaha/雅马哈 mgp24x 调音台', '广西幸全壮药有限公司 天眼查', '中国画二十家 (张立辰 姜宝林 张仃 郭石夫 陈少珊 刘', '云服务器和独立服务器之前存在哪些差异?', '供应农商银行开放式柜台', '共77 件鬼虎冢运动鞋男相关商品', '【7月27 28日自驾游】去黄金洞寻宝,在地心里漂流', '古早万丹图片 第1760张', '三 弧顺衣身侧缝线;根据图中示例的角度弧顺分割位置.', 'jx5044xyzmf2江铃全顺牌邮政车图片|中国汽车网 汽车', '罗翔刑法理论卷 真题卷】现货先发厚大法考2021年罗翔', '【顺丰包邮】华朴上品 广西融安滑皮金桔新鲜水果桔子', '电线防水中间接头bht1.25热缩连接管绝缘冷压接线端子', '七言对联大全千古绝对', '市委书记陈昌旭在中界镇河坎村生态茶套种辣椒项目基地调研.', '拍3盒发4盒】艾丽 奥利司他胶囊 用于肥胖或体重超重患者q', '高温烤漆房工业烤箱高温固化箱环保高温房塑粉烘干房', ':雪乡内景.', '破世界纪录张常鸿夺得射击男子50米步枪三姿金牌', '3,等差数列和等比数列的基本运算【景云】.doc', '星巴克598型星奕月饼礼盒 抹茶玄米月饼', '山楂苹果糖水_的做法步骤:7', '苹果摆盘分步图解的做法图解4', '解放j6自喷漆火焰红色j6p汽车漆划痕修复补漆笔油漆富贵红咖啡金', '吃知了猴究竟有益还是有害赶紧来了解一下吧', 'led发光白鹭造型灯 大型公园玻璃钢仿真动物雕塑美陈', '德国lenze 伺服器.商品大图', '各电视台标志头像 安装截图', '英雄101钢笔【连盒一起拍】', '作为中国人工智能和人形机器人企业的代表彰显了新中国的科技创新成果', '世界头例拱形市场 公寓 停车场组合 荷兰鹿特丹拱形大市场', '【聆听,红色印记】八路军,新四军驻沪办事处旧址', '初学者练瑜伽,髋部太僵硬打不开,怎么办?_换边', '清代皇帝御赐"恩赏"双龙铜质鎏银奖牌', '学大庆鼓干劲开展社会主义劳动竞赛', '莱维妮等欧美女包品牌引爆国民奢侈品潮流', '2015届金融学专业毕业论文选题参考目录.doc', '辽宁省丹东市东港市振兴街69号', '因经营需要的人民币信托贷款合同协议书范本模板.docx', 'prolene线在胆总管空肠roux—en—y吻合术的临床研究', '▼长轴部分采用卫星轴设计,卫星轴上有上油做润滑,提高敲击时的顺畅', '山东鲁恒|手扶式单轮柴油压路机', '朵朵贝儿婴儿洗护套装新生儿洗护用品婴儿护肤品儿童宝宝洗浴用品', '实木地板橡木仿古纯实木地板 室内木地板厂家直销特价', '软麦饭石蛭石绿沸石火山岩珍珠岩硅藻土陶粒球多肉介质营养铺面土', '矢量手绘手托花朵', '新款户外组合铁艺花箱园艺花槽商业街外摆花器隔断花坛长方形abcd', '蔓妙 秋冬短靴子时尚保暖防滑短靴中跟粗跟', '小口径户用超声波热量表日常维护需知', '肉鸽的适应气温为4℃ 40℃,因此在我国各个地区均可饲养.', '2018年兰州交通大学世界排名,中国排名,专业排名,现有', '新款夹棉长袖碎花衬衫女中长款价格质量 哪个牌子比较', '博马娱乐城官网:揽胜 2011款 5.0 v8 na hse', '大沙地坑头一巷龒 龒号小区 龤室龤厅龤卫', '孜孜以求 老徐的硬笔书法练习', '河南省陕县地坑院旅游2016.4.17', '凉拌黄瓜西红柿,不一样的做法吃法,更加入味', '广为大众所知的萌物"皮卡丘"给出的答案是900亿美元.2', '民国手工朱砂红陶制农民大海碗胶东民俗生活用具摆件民俗摆件包老', '三明活动板房直销', '有哪位大神知道这种包装的冬虫夏草香烟是真还是假']
    for q in qlist:
        chat(q, model, tokenizer, max_length=args.max_length, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k)
