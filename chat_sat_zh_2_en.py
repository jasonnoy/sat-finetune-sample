import torch
from sat import AutoModel
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy
from sat.model.official import ChatGLM2Model
from sat.model.finetune.lora2 import LoraMixin
from tqdm import tqdm


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
    parser.add_argument("--temperature", type=float, default=0.01)
    args = parser.parse_args()

    # load model
    # model, model_args = FineTuneModel.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690270701725/checkpoints/finetune-chatglm-6b-lora-07-25-15-39', 
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690281036622/checkpoints/finetune-chatglm2-6b-07-25-18-31',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690278447134/checkpoints/finetune-chatglm-6b-lora-07-25-17-48',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690281747909/checkpoints/finetune-chatglm2-6b-07-25-18-53',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/dm/snapbatches/1690296760278/checkpoints/finetune-chatglm2-6b-07-25-22-53',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/jjh/projects/ftsample/checkpoints/finetune-chatglm2-6b-07-27-10-19',
    # model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/jjh/projects/ftsample/checkpoints/finetune-chatglm2-6b-07-28-10-26',
    model, model_args = ChatGLM2Model.from_pretrained('/nxchinamobile2/shared/jjh/projects/sat-finetune-sample/checkpoints/finetune-chatglm2-6b-08-07-11-23',
    args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True,
    ))
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
    
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    qlist = [
            'Pro Taper Rear Sprocket for 2008 KTM 505SXF Sunstar Aluminum Rear Sprocket. ', 'Illustration of facades in Paris. ', "Hand-engraved baccarat paperweight, baccarat crystal, cherry blossom, heart-shaped paperweight, collector's gift for women. ", 'The price of iron ore has plunged from $120 a tonne 12 months ago to $63 tonne a tonne. ', 'Dancers performing on stage from the Creative Arts Dance Academy during Andover Day in downtown Andover. ', 'A&P Christmas Vinyl Record Albums transferred to CD. ', '"I Want You To Give Me A Discount Tote Bag" ', 'A photo of Palm Beach Gardens, FL, with a Latin band performing. ', 'Maytag 30" Built-In Single Electric Convection Wall Oven in Stainless steel. ', 'JCB 8014 CTS MINI DIGGER, YEAR 2016, with 935 hours, complete with digging bucket. ', 'A crop of a girl wearing yellow protective rubber gloves, wiping furniture. ', 'Private issue SAHARA TAHOE Lounge Act comedian signed late 70s. ', "Men's Knight Left Hand Golf Clubs with Bag & Cart Belleville Belleville Area image. ", 'Free shipping on 2*30cm*100cm 11color Auto Flash Point Film Tail light Tint Vinyl Film sticker. ', 'Turbofan E23D3/2C - Half Size Digital Electric Convection Ovens with a Stainless Steel Base Stand. ', 'Google Algorithm Updates and Changes infographic about social media. ', 'Dr. Bonnie Henry, regional health officer for B.C., discussing the potential impact of the mandatory mask order. ', "Women's cute fruit food socks for coffee, avocado, apple, and cherry Hamburger eggs. ", 'Angel Sleeve Blouse styles and fashion inspiration. ', "Transparent door stopper buffer for furniture protection in children's bedroom and kitchen. ", 'Premium felt cat cave bed for large cats and kittens. ', 'Electro Essence 30ml ABFE from Broome Natural Wellness. ', 'Blue beach umbrellas at Point Of Rocks, Crescent Beach, Siesta Key. ', 'Peach Indian Jacquard Brocade Wedding Dress Fabric. ', '"The Inventor and the Tycoon: The Murderer Eadweard Muybridge, the Entrepreneur Leland Stanford, and the Birth of Moving Pictures" book. ', 'Thermaltake Radiator Pacific CL480 (480mm, 5x G 1/4, black metal). ', 'A juvenile striped skunk captured in Lexington, KY. ', 'Kobe iPad 2 stand case with port designs. ', 'Single Family Home For Sale: 6363 Cedar Sage Trail. ', 'Breakfast pizza with cheddar, kale, and bacon. ', 'Not a caption', 'After Mein Kampf (1940) and Here Is Germany. ', 'Double outdoor spot-luminaire Helia LED IP55 H10 cm-white outdoor wall lamp. ', 'Jesus Christ crucified on the cross. ', '"1992 Country Christmas: An Old Fashioned Christmas" hardcover book with jacket. ', 'The property located at 106 N 181st St in Shoreline was sold for $745,000. I represented the buyer. ', 'Kiddy Guardianfix Pro 2-3 car seat in Racing Black. ', 'INFINITE 2013 calendar wallpaper collection. '
            ]

    for q in tqdm(qlist):
        chat(q, model, tokenizer, max_length=args.max_length, num_beams=args.num_beams, top_p=args.top_p, temperature=args.temperature, top_k=args.top_k)
