# JTP-Discord-Mod-Bot
JTP-Discord-Mod-Bot


#Get token
[https://discord.com/developers/docs/intro](https://discord.com/developers/applications)
new application
bot tab > reset token (copy save token for set)
turn on Server Members Intent
turn on Message Content Intent



#setup

python -m venv .venv

.\.venv\Scripts\activate

pip install -r ./requirements.txt

dotenv set BOT_TOKEN YOUR_TOKEN_HERE

dotenv set BOTPOSTCHANNEL YOUR_CHANNEL_NUMBER_HERE

dotenv set TAG_BLOCK_LIST {'tag1':.02,'tag2':.3}




#run

.\.venv\Scripts\activate

python .\JTP_bot.py
