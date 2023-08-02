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
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/jjh/projects/ftsample/checkpoints/finetune-chatglm2-6b-07-28-10-26',
    model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/jjh/projects/sat-finetune-sample/checkpoints/finetune-chatglm2-6b-08-02-19-08',
    args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True,
    ))
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    qlist = [
    "Pro Taper Rear Sprocket for 2008 KTM 505SXF Sunstar Aluminum Rear Sprocket.",
        "illustration of facades in paris stock vector - 11956818",
        "hand engraved baccarat paperweight, baccarat crystal, cherry blossom, heart paper weight, collector gifts, flowers, mothers day gift",
        "the price of iron ore has plunged from $120 a tonne 12 months ago to $63 tonne a tonne aap",
        "TIM JEAN/Staff photo<br /> Dancers perform on stage from the Creative Arts Dance Academy during Andover Day downtown Andover. 9/10/16",
        "A&P Christmas Vinyl Record Albums transferred to CD",
        "i want you to give me a discount tote bags",
        "florida latin beat | palm beach gardens, fl | latin band | photo 16",
        '"Maytag - 30"" Built-In Single Electric Convection Wall Oven - Stainless steel - Larger Front"',
        "JCB 8014 CTS MINI DIGGER, YEAR 2016, ONLY 935 HOURS, COMPLETE WITH DIGGING BUCKET *PLUS VAT*",
        "Crop of girl wearing in yellow protective rubber gloves, wiping furniture. royalty free stock image",
        "DANNY MARONA Live Private Issue SAHARA TAHOE Lounge Act COMEDIAN Signed Late 70s",
        "Men's Knight Left Hand Golf Clubs with bag & Cart Belleville Belleville Area image 1",
        "Free shipping 2*30cm*100cm 11color Auto Flash Point Film Tail light Tint Vinyl Film sticker",
        "Turbofan E23D3/2C - Half Size Digital Electric Convection Ovens Double Stacked on a Stainless Steel Base Stand",
        "Google Algorithm Updates and Changes 1998-2012 Infographic | AtDotCom Social media | Scoop.it",
        "Provincial health officer Dr. Bonnie Henry says B.C.'s mandatory mask order could be gone as soon as July as COVID-19 infection rates fall and vaccination rises. (B.C. government)",
        "Women Cute Fruit Food Socks Coffee Avocado Apple Cherry Hamburger Egg Donutsdresskily-dresskily",
        "Angel Sleeve Blouse Source by wulantriasStyleOnme_Minimal Angel Sleeve Blouse Source by wulantrias Styleonme StyleOnme_Leaf Print Unique Neckline Blouse Boat Neck Tie-Waist Jumpsuit StyleOnme_Pleated Sleeve Peplum Blouse Tops Asian Fashion, Hijab Fashion, Fashion Dresses, Women's Fashion, New Blouse Designs, Mode Chic, Blouse Dress, Saree Blouse, Blouse Styles",
        "Door-Stopper Buffer Furniture Protection-Walls Children Transparent And Bedroom Kitchen",
        "MEOWFIA Premium Felt Cat Cave Bed (Large) - Eco Friendly 100% Merino Wool Bed for Large Cats and Kittens (Asphalt/Aquamarine)",
        "Electro Essence 30ml ABFE - Broome Natural Wellness",
        "Blue Beach Umbrellas, Point Of Rocks, Crescent Beach, Siesta Key - Spiral Notebook",
        "Peach Indian Jacquard Brocade Wedding Dress Fabric By The Yard | Etsy Floral Cushions, Floral Fabric, Brocade Fabric, Jacquard Fabric, Fabric Crafts, Sewing Crafts, Etsy Fabric, Gold Silk, Peach Colors",
        "img - for The Inventor and the Tycoon: The Murderer Eadweard Muybridge, the Entrepreneur Leland Stanford, and the Birth of Moving Pictures book / textbook / text book",
        "Thermaltake Radiator Pacific CL480 (480mm, 5x G 1/4, miedź) czarny",
        "A juvenile striped skunk captured in Lexington KY",
        "PORT DESIGNS Kobe iPad 2 stand case",
        "Flower Mound Single Family Home For Sale: 6363 Cedar Sage Trail",
        "Cracked egg on naan breakfast pizza with cheddar, kale and bacon",
        "Natalie Brooks Secrets of Treasure House large screenshot",
        "WWII - After Mein Kampf (1940) / Here Is Germany",
        "Outdoor wall lamp - Double outdoor spot-luminaire Helia LED IP55 H10 cm-white - SLV Belgium",
        "istock banner with Jesus Christ crucified on the cross 1203974480",
        "MBA #3939-161  ""1992 Country Christmas An Old Fashioned Christmas"" Hard Cover With Jacket""",
        '106 N 181st St | Shoreline  Sold for $745,000   Represented the Buyer',
        'siege auto kiddy guardian kiddy guardianfix pro 2 1 2 3 car seat racing black amazon',
        'INFINITE images infinite calendar 2013 wallpaper photos 33674573',
    "https://www.white-ibiza.com/wp-content/uploads/2020/03/ibiza-wedding-venues-amante-ibiza-2020-00-1025x1536.jpg",
    "Garlic_bread_sticks - 55545390_1904536586323402_8138537957301682176_n.jpg",
    "05f2e9e8751 Fila - Ray Low Wmn white-silver - 00014201669374 - s...",
    "🌟MPV FOR RENT FROM $380*WEEKLY🌟",
    "Seriously_Sabrina_Photography_Dayton_Ohio_Great_Gatsby_Wedding_Cedar_Springs_Pavillion_Harper_78.jpg",
    "Rustic Outdoor Blue Welcome Outdoor Welcome Sign by ...",
    "TXT (투모로우바이투게더) - Our Summer (Color Coded Lyrics Eng/Rom/Han/가사)"
            ]
    for q in qlist:
        chat(q, model, tokenizer, max_length=args.max_length, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k)
