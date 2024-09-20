# JTP-Discord-Mod-Bot
JTP-Discord-Mod-Bot

### Get BOT token 
1. [https://discord.com/developers/docs/intro](https://discord.com/developers/applications)
2. new application
3. bot tab > reset token (copy save token for set)
4. turn on Server Members Intent
5. turn on Message Content Intent

### setup
1. python -m venv .venv
2. .\.venv\Scripts\activate
3. pip install -r ./requirements.txt
4. dotenv set BOT_TOKEN YOUR_TOKEN_HERE
5. dotenv set BOTPOSTCHANNEL YOUR_CHANNEL_NUMBER_HERE
6. dotenv set TAG_BLOCK_LIST {'tag1':.02,'tag2':.3}
7. dotenv set SFW_BLOCK_LIST {'tag1':.02,'tag2':.3}
8. dotenv set NSWF_BLOCK_LIST {'tag1':.02,'tag2':.3}
9. dotenv set FEMALE_BLOCK_LIST {'tag1':.02,'tag2':.3}
10. dotenv set MALE_BLOCK_LIST {'tag1':.02,'tag2':.3}
11. dotenv set STRAIGHT_BLOCK_LIST {'tag1':.02,'tag2':.3}
12. dotenv set FERAL_BLOCK_LIST {'tag1':.02,'tag2':.3}
13. dotenv set HUMAN_BLOCK_LIST {'tag1':.02,'tag2':.3}
14. dotenv set SPOILER_BLOCK_LIST {'tag1':.02,'tag2':.3}

### run
1. .\.venv\Scripts\activate
2. python .\JTP_bot.py
