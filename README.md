# JTP-Discord-Mod-Bot
JTP-Discord-Mod-Bot


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
