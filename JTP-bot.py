# Requires the 'members' and 'message_content' privileged intents to function.
# Requires posting to send messages
# If you enable deleting, then it requires that permtion as well.

#discord bot
import discord
from discord.ext import commands
import random
import io

# loading images from URL
import requests

#autotagger
import json

import gradio as gr
from PIL import Image
import safetensors.torch
import timm
from timm.models import VisionTransformer
import torch
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

#url verification
from urllib.parse import urlparse

#download model from hugging face
from huggingface_hub import hf_hub_download


#load BOT_TOKEN as enviroment variable
import os
import ast 
from dotenv import load_dotenv
load_dotenv()


#global settings

#autotagger
#min THRESHOLD, to drop most tags not used
THRESHOLD = 0.1

#set the channel the bot posts messages in from environmental variables
botPostChannel = os.environ["BOTPOSTCHANNEL"]
print(botPostChannel)

#set the autotagger tags to watchfor from environmental variables
#example list
#IE pony:.2 would be respond to images that give a autotagger score that includes pony 20% or higher.
#tag_block_list = {'pony':.02,'equid':.3}

#tags to block in all channels
tag_block_list = ast.literal_eval(os.environ["TAG_BLOCK_LIST"])

print("TAG_BLOCK_LIST is:")
for block in tag_block_list:
    print(f'Tag to block {block}: {tag_block_list[block]}')


#tags to only block in SFW channels
sfw_block_list = ast.literal_eval(os.environ["SFW_BLOCK_LIST"])

print("sfw_block_list is:")
for sfw in sfw_block_list:
    print(f'Tag to block {sfw}: {sfw_block_list[sfw]}')


#tags to only block in SFW channels
nsfw_block_list = ast.literal_eval(os.environ["NSWF_BLOCK_LIST"])

print("nsfw_block_list is:")
for nsfw in nsfw_block_list:
    print(f'Tag to block {nsfw}: {nsfw_block_list[nsfw]}')


#tags to only block in female channels
female_block_list = ast.literal_eval(os.environ["FEMALE_BLOCK_LIST"])

print("female_block_list is:")
for female in female_block_list:
    print(f'Tag to block {female}: {female_block_list[female]}')

#tags to only block in male channels
male_block_list = ast.literal_eval(os.environ["MALE_BLOCK_LIST"])

print("male_block_list is:")
for male in male_block_list:
    print(f'Tag to block {male}: {male_block_list[male]}')

#tags to only block in straight channels
straight_block_list = ast.literal_eval(os.environ["STRAIGHT_BLOCK_LIST"])

print("straight_block_list is:")
for straight in straight_block_list:
    print(f'Tag to block {straight}: {straight_block_list[straight]}')

#tags to only block in straight channels
feral_block_list = ast.literal_eval(os.environ["FERAL_BLOCK_LIST"])

print("feral_block_list is:")
for feral in feral_block_list:
    print(f'Tag to block {feral}: {feral_block_list[feral]}')



#details
description = '''A discord bot, to run posted images through autotagger for moderation. 

I am not affiliated with autotagger model in anyway.
https://huggingface.co/spaces/cdnuts/JointTaggerProject-Inference-Beta-GPU/tree/main
from
https://huggingface.co/cdnuts
This tagger used by this discord bot the result of joint efforts between members of the RedRocket team, with distinctions given to Thessalo for creating the foundation for this project with his efforts, RedHotTensors for redesigning the process into a second-order method that models information expectation, and drhead for dataset prep, creation of training code and supervision of training runs.
'''


# Download dataset
print("check for model")

import os.path
if ( os.path.isfile(".\\JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors") == False ):
    print("downloading JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors")
    hf_hub_download(repo_id="cdnuts/JointTaggerProject-Inference-Beta-GPU", filename="JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors", repo_type="space", local_dir=".\\")
else:
    print("JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors found")

if ( os.path.isfile(".\\tagger_tags.json") == False ):
    print("downloading tagger_tags.json")
    #print(
    hf_hub_download(repo_id="cdnuts/JointTaggerProject-Inference-Beta-GPU", filename="tagger_tags.json", repo_type="space", local_dir=".\\")
else:
    print("tagger_tags.json found")


#https://huggingface.co/spaces/cdnuts/JointTaggerProject-Inference-Beta-GPU/resolve/main/tagger_tags.json?download=true
#https://huggingface.co/cdnuts/JointTaggerProject-Inference-Beta-GPU/resolve/main/tagger_tags.json.

print("model check ended")

#discord bot
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix='?', description=description, intents=intents)


#url validator funtions
def url_valid(url):
    prased_url = urlparse(url)

    #verify url is https
    if ( prased_url.scheme != "https"):
        return False

    #verify discord CDN address 
    if ( prased_url.hostname != "cdn.discordapp.com" ):
        return False

    return True


#autotagger funtions


torch.set_grad_enabled(False)

class Fit(torch.nn.Module):
    def __init__(
        self,
        bounds: tuple[int, int] | int,
        interpolation = InterpolationMode.LANCZOS,
        grow: bool = True,
        pad: float | None = None
    ):
        super().__init__()

        self.bounds = (bounds, bounds) if isinstance(bounds, int) else bounds
        self.interpolation = interpolation
        self.grow = grow
        self.pad = pad

    def forward(self, img: Image) -> Image:
        wimg, himg = img.size
        hbound, wbound = self.bounds

        hscale = hbound / himg
        wscale = wbound / wimg

        if not self.grow:
            hscale = min(hscale, 1.0)
            wscale = min(wscale, 1.0)

        scale = min(hscale, wscale)
        if scale == 1.0:
            return img

        hnew = min(round(himg * scale), hbound)
        wnew = min(round(wimg * scale), wbound)

        img = TF.resize(img, (hnew, wnew), self.interpolation)

        if self.pad is None:
            return img

        hpad = hbound - hnew
        wpad = wbound - wnew

        tpad = hpad // 2
        bpad = hpad - tpad

        lpad = wpad // 2
        rpad = wpad - lpad

        return TF.pad(img, (lpad, tpad, rpad, bpad), self.pad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(" +
            f"bounds={self.bounds}, " +
            f"interpolation={self.interpolation.value}, " +
            f"grow={self.grow}, " +
            f"pad={self.pad})"
        )

class CompositeAlpha(torch.nn.Module):
    def __init__(
        self,
        background: tuple[float, float, float] | float,
    ):
        super().__init__()

        self.background = (background, background, background) if isinstance(background, float) else background
        self.background = torch.tensor(self.background).unsqueeze(1).unsqueeze(2)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-3] == 3:
            return img

        alpha = img[..., 3, None, :, :]

        img[..., :3, :, :] *= alpha

        background = self.background.expand(-1, img.shape[-2], img.shape[-1])
        if background.ndim == 1:
            background = background[:, None, None]
        elif background.ndim == 2:
            background = background[None, :, :]

        img[..., :3, :, :] += (1.0 - alpha) * background
        return img[..., :3, :, :]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(" +
            f"background={self.background})"
        )

transform = transforms.Compose([
    Fit((384, 384)),
    transforms.ToTensor(),
    CompositeAlpha(0.5),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    transforms.CenterCrop((384, 384)),
])

model = timm.create_model(
    "vit_so400m_patch14_siglip_384.webli",
    pretrained=False,
    num_classes=9083,
) # type: VisionTransformer

class GatedHead(torch.nn.Module):
    def __init__(self,
        num_features: int,
        num_classes: int
    ):
        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes * 2)

        self.act = torch.nn.Sigmoid()
        self.gate = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.act(x[:, :self.num_classes]) * self.gate(x[:, self.num_classes:])
        return x

model.head = GatedHead(min(model.head.weight.shape), 9083)

safetensors.torch.load_model(model, "JTP_PILOT2-e3-vit_so400m_patch14_siglip_384.safetensors")

if torch.cuda.is_available():
    model.cuda()
    if torch.cuda.get_device_capability()[0] >= 7: # tensor cores
        model.to(dtype=torch.float16, memory_format=torch.channels_last)

model.eval()

with open("tagger_tags.json", "r") as file:
    tags = json.load(file) # type: dict
allowed_tags = list(tags.keys())

#for idx, tag in enumerate(allowed_tags):
    #allowed_tags[idx] = tag.replace("_", " ")

sorted_tag_score = {}

def run_classifier(image, threshold):
    global sorted_tag_score
    img = image.convert('RGBA')
    tensor = transform(img).unsqueeze(0)

    if torch.cuda.is_available():
        tensor = tensor.cuda()
        if torch.cuda.get_device_capability()[0] >= 7: # tensor cores
            tensor = tensor.to(dtype=torch.float16, memory_format=torch.channels_last)

    with torch.no_grad():
        probits = model(tensor)[0].cpu()
        values, indices = probits.topk(250)

    tag_score = dict()
    for i in range(indices.size(0)):
        tag_score[allowed_tags[indices[i]]] = values[i].item()
    sorted_tag_score = dict(sorted(tag_score.items(), key=lambda item: item[1], reverse=True))

    return create_tags(threshold)

def create_tags(threshold):
    global sorted_tag_score
    filtered_tag_score = {key: value for key, value in sorted_tag_score.items() if value > threshold}
    text_no_impl = ", ".join(filtered_tag_score.keys())
    return text_no_impl, filtered_tag_score


#Download Model



#discord bot code

#prossess messages
#ignore bot's own messages
#ignore DM messages
#prosses messages for attachments.
image_types = ["png", "jpeg", "gif", "jpg", "webp"] #image formats supported by discord not mov, mpeg, apng, 
@bot.event
async def on_message(message: discord.Message):
    # this is the sender of the Message
    #print(bot.user.id)
    #print(message.author)
    #print(message.author.id)

    #ingore messages posted by bot to avoid infinite loop
    if message.author.id == bot.user.id:
        return

    #ingore messages sent by Direct Message to avoid spam
    if isinstance(message.channel, discord.DMChannel):
        return

    issue = 0
    issues = ""

    #attachments
    for attachment in message.attachments:
        if any(attachment.filename.lower().endswith(image) for image in image_types):
            print(message.author)
            print(bot.user.id)

            print(f"[{message.guild}] | [{message.channel}] | [{message.author}] @ {message.created_at}: {message.content}")

            #print(attachment.url)
            #print(url_valid(attachment.url))
            if(url_valid(attachment.url)):

                print(f'Attachment {attachment.filename}.')
                #load file
                image = requests.get(attachment.url, stream=True)
                if (image.status_code == 200):

                    #resize image
                    base_width = 512
                    img = Image.open(requests.get(attachment.url, stream=True).raw)
                    wpercent = (base_width / float(img.size[0]))
                    hsize = int((float(img.size[1]) * float(wpercent)))
                    img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)

                    #run autotagger on image
                    scores = run_classifier(img,0.1)[1]

                    #print(scores)

                    #review tags for block tags
                    for score in scores:
                        if scores[score] > THRESHOLD:
                            print(f"{score}: {str(round((scores[score]*100)))}%")

                            for block in tag_block_list:
                                #if tags match
                                if score == block:
                                    print("matched")
                                    #if tag's score is greater then the specified score for tag in block list 
                                    if scores[score] >= tag_block_list[block]:
                                        print("[ * * * TAG_BLOCK_LIST * * * ]")
                                        issue += 1
                                        issues += "tags-> " + str(round((scores[score]*100))) + "% **" + score + "**  "

                            if "sfw" in str(message.channel) and "nsfw" not in str(message.channel):
                                #if ( message.channel.lower.find("sfw") ):
                                #if "sfw" in str(message.channel):
                                #if ( str(message.channel).find("sfw") != -1 ):
                                for sfw in sfw_block_list:
                                    if score == sfw:
                                        #if tag's score is greater then the specified score for tag in block list 
                                        if scores[score] >= sfw_block_list[sfw]:
                                            print("[ * * * SFW_BLOCK_LIST * * * ]")
                                            issue += 1
                                            issues += "SFW " + str(round((scores[score]*100))) + "% **" + score + "**  "
                            else:
                                for nsfw in nsfw_block_list:
                                    if score == nsfw:
                                        print("matched")
                                        #if tag's score is greater then the specified score for tag in block list 
                                        if scores[score] >= nsfw_block_list[nsfw]:
                                            print("[ * * * NSFW_BLOCK_LIST * * * ]")
                                            issue += 1
                                            issues += "NSFW-> " + str(round((scores[score]*100))) + "% **" + score + "**  "

                            if "female" in str(message.channel):
                                for female in female_block_list:
                                    if score == female:
                                        print("matched")
                                        #if tag's score is greater then the specified score for tag in block list 
                                        if scores[score] >= female_block_list[female]:
                                            print("[ * * * FEMALE_BLOCK_LIST * * * ]")
                                            issue += 1
                                            issues += "FEMALE-> " + str(round((scores[score]*100))) + "% **" + score + "**  "

                            if "male" in str(message.channel):
                                for male in male_block_list:
                                    if score == male:
                                        print("matched")
                                        #if tag's score is greater then the specified score for tag in block list 
                                        if scores[score] >= male_block_list[male]:
                                            print("[ * * * MALE_BLOCK_LIST * * * ]")
                                            issue += 1
                                            issues += "MALE-> " + str(round((scores[score]*100))) + "% **" + score + "**  "

                            if "straight" in str(message.channel):
                                for straight in straight_block_list:
                                    if score == straight_block_list:
                                        print("matched")
                                        #if tag's score is greater then the specified score for tag in block list 
                                        if scores[score] >= straight_block_list[straight]:
                                            print("[ * * * STRAIGHT_BLOCK_LIST * * * ]")
                                            issue += 1
                                            issues += "STRAIGHT-> " + str(round((scores[score]*100))) + "% **" + score + "**  "

                            if "feral" not in str(message.channel):
                                for feral in feral_block_list:
                                    if score == feral_block_list:
                                        print("matched")
                                        #if tag's score is greater then the specified score for tag in block list 
                                        if scores[score] >= feral_block_list[feral]:
                                            print("[ * * * FERAL_BLOCK_LIST * * * ]")
                                            issue += 1
                                            issues += "FERAL-> " + str(round((scores[score]*100))) + "% **" + score + "**  "            

                else:
                    print("https request error")
            else:
                print("url invalid")


    if issue > 0:
        print("][ * * * BLOCK_LIST  * * * ][")
        await bot.get_channel(int(botPostChannel)).send(f"{message.author.mention} posted {message.jump_url}, image(s) {issues}")
        #await message.reply(f"<@{message.author.id}> your message was deleted; because AI thinks there was a chance that one, or more images in the post contained something against the rules. {issues}")
        #await message.delete()
	
    await bot.process_commands(message)


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')


#set the BOT_TOKEN from environmental variables
bot.run(os.environ["BOT_TOKEN"])