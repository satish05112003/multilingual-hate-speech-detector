"""
CivicGuard AI — Dataset Collection Module
Downloads and merges multilingual hate speech datasets.
"""

import os
import pandas as pd
import requests
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "datasets"
DATA_DIR.mkdir(exist_ok=True)


def create_english_dataset():
    """
    Create English hate speech dataset from manually curated samples.
    Inspired by Davidson + Jigsaw datasets — categories: hate, offensive, neutral.
    """
    print("[EN] Creating English dataset...")

    samples = [
        # === HATE SPEECH ===
        ("I hate all immigrants, they should be deported", "hate"),
        ("These people are subhuman and don't deserve rights", "hate"),
        ("All [ethnic group] are terrorists and criminals", "hate"),
        ("They should be exterminated from this country", "hate"),
        ("Go back to your country you filthy animal", "hate"),
        ("People like you deserve to die", "hate"),
        ("Your kind is a disease to society", "hate"),
        ("We need to cleanse our nation of these parasites", "hate"),
        ("You're nothing but a worthless piece of trash because of your race", "hate"),
        ("All of them should be locked up and thrown away the key", "hate"),
        ("These people are genetically inferior", "hate"),
        ("I wish they would all just disappear from the earth", "hate"),
        ("The world would be better without your kind", "hate"),
        ("You people are animals, not humans", "hate"),
        ("Death to all of them, every single one", "hate"),
        ("They are invading our country like cockroaches", "hate"),
        ("Your religion is evil and should be banned", "hate"),
        ("People of your skin color are all criminals", "hate"),
        ("You don't belong here, go die somewhere else", "hate"),
        ("We should build walls to keep these vermin out", "hate"),
        ("All women belong in the kitchen, they're useless otherwise", "hate"),
        ("Gay people are disgusting and deserve punishment", "hate"),
        ("Disabled people are a burden on society", "hate"),
        ("Your culture is barbaric and primitive", "hate"),
        ("I want to see every one of them suffer", "hate"),
        ("These refugees are ruining our beautiful country", "hate"),
        ("People with dark skin are inherently violent", "hate"),
        ("Trans people are mentally ill freaks", "hate"),
        ("Old people should just die and stop wasting resources", "hate"),
        ("Jews control the media and the banks", "hate"),
        ("Muslims are all potential terrorists", "hate"),
        ("Black people are more prone to crime genetically", "hate"),
        ("Women shouldn't be allowed to vote", "hate"),
        ("Asian people are all the same, no individuality", "hate"),
        ("Indigenous people are savages who couldn't build civilization", "hate"),
        ("Fat people are disgusting and have no self control", "hate"),
        ("Your accent makes you sound like a monkey", "hate"),
        ("All homeless people are drug addicts by choice", "hate"),
        ("People with mental illness should be institutionalized permanently", "hate"),
        ("Immigrants steal jobs and spread disease", "hate"),
        ("Mixed race children are abominations", "hate"),
        ("Your people contributed nothing to civilization", "hate"),
        ("They breed like animals and should be sterilized", "hate"),
        ("Autistic people shouldn't be in regular schools", "hate"),
        ("Poor people are poor because they're lazy and stupid", "hate"),
        ("We need ethnic cleansing in this country", "hate"),
        ("That community is full of rapists", "hate"),
        ("These foreigners are polluting our gene pool", "hate"),
        ("Your traditions are backward and uncivilized", "hate"),
        ("Certain races are just naturally less intelligent", "hate"),

        # === OFFENSIVE ===
        ("You're such an idiot, can't believe you said that", "offensive"),
        ("What a stupid comment, think before you speak", "offensive"),
        ("Shut up you moron", "offensive"),
        ("This is the dumbest thing I've ever heard", "offensive"),
        ("You're so annoying, just go away", "offensive"),
        ("What a loser, get a life", "offensive"),
        ("Stop being such a crybaby about everything", "offensive"),
        ("You're a joke, nobody takes you seriously", "offensive"),
        ("God you're pathetic", "offensive"),
        ("Can you just shut your mouth for once", "offensive"),
        ("This exam is killing me right now", "offensive"),
        ("That movie was so trash I want my time back", "offensive"),
        ("You look terrible in that outfit", "offensive"),
        ("Your cooking is absolutely disgusting", "offensive"),
        ("I can't stand people like you who whine all the time", "offensive"),
        ("What a waste of space", "offensive"),
        ("You're delusional if you think that's true", "offensive"),
        ("That's the ugliest thing I've ever seen", "offensive"),
        ("Nobody asked for your worthless opinion", "offensive"),
        ("You're embarrassing yourself right now", "offensive"),
        ("This traffic is making me want to kill myself", "offensive"),
        ("I'm so done with these clowns in management", "offensive"),
        ("What a dumpster fire of a presentation", "offensive"),
        ("That was the most boring lecture ever, professor sucks", "offensive"),
        ("Your taste in music is absolute garbage", "offensive"),
        ("I hate this weather so much, it's disgusting", "offensive"),
        ("This food tastes like cardboard, worst restaurant ever", "offensive"),
        ("You drive like a maniac, are you blind?", "offensive"),
        ("This code is so bad it gave me a headache", "offensive"),
        ("The customer service here is absolutely terrible", "offensive"),
        ("Are you deaf or just choosing to ignore me?", "offensive"),
        ("Get lost you weirdo", "offensive"),
        ("You have zero talent, stop embarrassing yourself", "offensive"),
        ("What a clown show this meeting was", "offensive"),
        ("Your handwriting looks like a chicken walked on paper", "offensive"),
        ("I swear this app was made by monkeys", "offensive"),
        ("Can't believe they let someone this clueless be in charge", "offensive"),
        ("This is bullshit, pure and simple", "offensive"),
        ("You're acting like a total brat right now", "offensive"),
        ("My boss is the worst human being alive", "offensive"),
        ("This assignment is absolute hell", "offensive"),
        ("You people have no common sense whatsoever", "offensive"),
        ("I could punch this computer right now", "offensive"),
        ("What a nightmare of a day", "offensive"),
        ("You clearly have no idea what you're talking about", "offensive"),
        ("This whole system is broken and useless", "offensive"),
        ("Whoever designed this UI should be fired", "offensive"),
        ("That joke was so bad it physically hurt", "offensive"),
        ("Stop wasting everyone's time with your nonsense", "offensive"),
        ("You're the definition of incompetence", "offensive"),

        # === NEUTRAL ===
        ("I love spending time with my family on weekends", "neutral"),
        ("The weather today is really beautiful", "neutral"),
        ("Just finished reading an amazing book", "neutral"),
        ("Life is too beautiful to waste on negativity", "neutral"),
        ("Everyone deserves to be treated with respect", "neutral"),
        ("Let's work together to make the world a better place", "neutral"),
        ("Happy birthday! Wishing you the best", "neutral"),
        ("Just had a great meal at the new restaurant", "neutral"),
        ("The sunset looks absolutely stunning today", "neutral"),
        ("I believe in equality for all people", "neutral"),
        ("Education is the key to a better future", "neutral"),
        ("Stay strong, you can get through this difficult time", "neutral"),
        ("Every culture has something beautiful to offer", "neutral"),
        ("I appreciate the diversity in our community", "neutral"),
        ("Great job on the presentation, very impressive work", "neutral"),
        ("This new movie has an incredible storyline", "neutral"),
        ("Kindness costs nothing but means everything", "neutral"),
        ("Learning a new language opens up so many opportunities", "neutral"),
        ("The park is so peaceful early in the morning", "neutral"),
        ("I'm grateful for all the good in my life", "neutral"),
        ("The concert last night was absolutely phenomenal", "neutral"),
        ("Just planted some new flowers in the garden", "neutral"),
        ("My dog learned a new trick today and I'm so proud", "neutral"),
        ("Reading about history helps us understand the present", "neutral"),
        ("This coffee shop has the coziest ambiance", "neutral"),
        ("The new policy benefits everyone equally", "neutral"),
        ("Just finished a great workout, feeling energized", "neutral"),
        ("The kids at the school fundraiser were so enthusiastic", "neutral"),
        ("Good morning everyone, have a wonderful day ahead", "neutral"),
        ("The documentary about ocean life was fascinating", "neutral"),
        ("I will kill it on stage tonight with my performance", "neutral"),
        ("These prices are killing my budget this month", "neutral"),
        ("She slayed the competition in the debate", "neutral"),
        ("That comedian absolutely destroyed the audience with laughter", "neutral"),
        ("The test was brutal but I think I passed", "neutral"),
        ("I'm dying to try that new restaurant downtown", "neutral"),
        ("He killed it with that guitar solo, amazing talent", "neutral"),
        ("This deadline is murdering my sleep schedule", "neutral"),
        ("She's on fire today with her basketball skills", "neutral"),
        ("The crowd went wild when the team scored", "neutral"),
        ("Meditation has completely changed my perspective on life", "neutral"),
        ("Volunteering at the shelter is incredibly rewarding", "neutral"),
        ("The new library has an amazing collection of books", "neutral"),
        ("Supporting local businesses strengthens our community", "neutral"),
        ("Art has the power to bring people together", "neutral"),
        ("Teamwork makes the dream work, great collaboration today", "neutral"),
        ("The sunrise from the mountain top was breathtaking", "neutral"),
        ("Music transcends all boundaries and connects people", "neutral"),
        ("Just completed my first marathon, what an achievement", "neutral"),
        ("The children's laughter filled the entire playground", "neutral"),
    ]

    df = pd.DataFrame(samples, columns=["text", "label"])
    df.to_csv(DATA_DIR / "english_dataset.csv", index=False)
    print(f"  → Created {len(df)} English samples")
    return df


def create_hindi_hinglish_dataset():
    """
    Create Hindi / Hinglish hate speech dataset.
    Inspired by HASOC + Hinglish datasets.
    """
    print("[HI] Creating Hindi/Hinglish dataset...")

    samples = [
        # === HATE ===
        ("तुम जैसे लोगों को देश से निकाल देना चाहिए", "hate"),
        ("तुम्हारी जाति के लोग सब चोर हैं", "hate"),
        ("ये लोग इंसान नहीं जानवर हैं", "hate"),
        ("इनको मार डालना चाहिए सबको", "hate"),
        ("तुम्हारे धर्म के लोग आतंकवादी हैं", "hate"),
        ("ये सब कीड़े मकोड़े हैं समाज के", "hate"),
        ("तुम जैसे गंदे लोगों की कोई जगह नहीं", "hate"),
        ("इन लोगों को जिंदा जला देना चाहिए", "hate"),
        ("तुम्हारी जाति नीच है", "hate"),
        ("ये मुसलमान सब देशद्रोही हैं", "hate"),
        ("दलितों को कोई अधिकार नहीं मिलना चाहिए", "hate"),
        ("ये बिहारी लोग हमारे शहर गंदा कर रहे हैं", "hate"),
        ("औरतों की कोई अक्ल नहीं होती", "hate"),
        ("ये भिखारी लोग मर क्यों नहीं जाते", "hate"),
        ("तुम जैसे काले लोग बदसूरत होते हैं", "hate"),
        ("ये लोग गंदी नाली के कीड़े हैं", "hate"),
        ("इनका खून बहाना चाहिए", "hate"),
        ("तुम्हारा समुदाय कैंसर है समाज का", "hate"),
        ("ये foreigners को भगाओ यहाँ से", "hate"),
        ("LGBTQ लोग पागल और बीमार हैं", "hate"),

        # Hinglish hate
        ("tum jaise logo ko desh se nikaal dena chahiye", "hate"),
        ("tumhari caste ke log sab chor hain", "hate"),
        ("ye log insaan nahi janwar hain", "hate"),
        ("inko maar daalo saalo ko", "hate"),
        ("tum gande log ho, tumhari koi jagah nahi", "hate"),
        ("ye sab keede makode hain society ke", "hate"),
        ("tumhari community cancer hai society ka", "hate"),
        ("tum jaise kale log ugly hote hain", "hate"),
        ("ye bihari log hamare city ganda kar rahe hain", "hate"),
        ("muslims sab deshdrohii hain", "hate"),
        ("dalits ko koi rights nahi milne chahiye", "hate"),
        ("auraton ki koi akal nahi hoti", "hate"),
        ("ye beggars mar kyu nahi jaate", "hate"),
        ("inke khoon bahana chahiye", "hate"),
        ("LGBTQ people pagal aur beemar hain", "hate"),

        # === OFFENSIVE ===
        ("तू बहुत बड़ा बेवकूफ है यार", "offensive"),
        ("क्या बकवास कर रहा है तू", "offensive"),
        ("चुप हो जा गधे", "offensive"),
        ("तेरी शक्ल देखकर हंसी आती है", "offensive"),
        ("कितना गिरा हुआ इंसान है तू", "offensive"),
        ("तू तो एक नंबर का ढोंगी है", "offensive"),
        ("पागल है क्या तू पूरा", "offensive"),
        ("तुझसे बड़ा वाहियात कोई नहीं", "offensive"),
        ("ये गाना सुनकर कान से खून आ गया", "offensive"),
        ("ये खाना खाकर मर जाऊंगा", "offensive"),
        ("tu bahut bada bewkoof hai yaar", "offensive"),
        ("kya bakwas kar raha hai tu", "offensive"),
        ("chup ho ja gadhe", "offensive"),
        ("teri shakal dekh ke hansi aati hai", "offensive"),
        ("kitna gira hua insaan hai tu", "offensive"),
        ("tu toh ek number ka dhongi hai", "offensive"),
        ("pagal hai kya tu poora", "offensive"),
        ("tujhse bada wahiyat koi nahi", "offensive"),
        ("ye gaana sunke kaan se khoon aa gaya", "offensive"),
        ("ye khana khaake mar jaunga", "offensive"),
        ("boss ne aaj dimag kha liya", "offensive"),
        ("ye traffic ki wajah se sir phata ja raha", "offensive"),
        ("iss company mein kaam karna torture hai", "offensive"),
        ("kya chutiyapa hai ye system", "offensive"),
        ("saale ko kuch nahi aata kaam", "offensive"),

        # === NEUTRAL ===
        ("आज का दिन बहुत अच्छा है", "neutral"),
        ("परिवार के साथ समय बिताना बहुत अच्छा लगता है", "neutral"),
        ("सबको प्यार और सम्मान दो", "neutral"),
        ("जीवन बहुत सुंदर है", "neutral"),
        ("शिक्षा सबका अधिकार है", "neutral"),
        ("मिलकर काम करने से सब संभव है", "neutral"),
        ("आज खाना बहुत स्वादिष्ट बना", "neutral"),
        ("कल का मौसम बहुत अच्छा रहेगा", "neutral"),
        ("नई किताब पढ़कर बहुत अच्छा लगा", "neutral"),
        ("सबको बराबर का मौका मिलना चाहिए", "neutral"),
        ("aaj ka din bahut acha hai", "neutral"),
        ("family ke saath time spend karna bahut acha lagta hai", "neutral"),
        ("sabko pyaar aur samman do", "neutral"),
        ("jeevan bahut sundar hai", "neutral"),
        ("education sabka right hai", "neutral"),
        ("milke kaam karne se sab possible hai", "neutral"),
        ("aaj khana bahut tasty bana", "neutral"),
        ("kal ka weather bahut acha rahega", "neutral"),
        ("nayi book padhke bahut acha laga", "neutral"),
        ("sabko equal opportunity milni chahiye", "neutral"),
        ("ye movie dekho, bahut inspiring hai", "neutral"),
        ("morning walk pe jaana health ke liye acha hai", "neutral"),
        ("coding seekhna bahut fascinating hai", "neutral"),
        ("team ne aaj bahut acha perform kiya", "neutral"),
        ("festival ki shopping ho gayi, bahut maza aaya", "neutral"),
    ]

    df = pd.DataFrame(samples, columns=["text", "label"])
    df.to_csv(DATA_DIR / "hindi_hinglish_dataset.csv", index=False)
    print(f"  → Created {len(df)} Hindi/Hinglish samples")
    return df


def create_bengali_dataset():
    """
    Create Bengali hate speech dataset.
    Inspired by HASOC Bengali subset.
    """
    print("[BN] Creating Bengali dataset...")

    samples = [
        # === HATE ===
        ("তোদের মতো লোকদের দেশ থেকে বের করে দেওয়া উচিত", "hate"),
        ("তোর জাতের লোক সব চোর", "hate"),
        ("এরা মানুষ না, পশু", "hate"),
        ("এদের সবাইকে মেরে ফেলা উচিত", "hate"),
        ("তোর ধর্মের লোক সব সন্ত্রাসী", "hate"),
        ("এরা সমাজের পোকা মাকড়", "hate"),
        ("তোদের মতো নোংরা লোকের কোনো জায়গা নেই", "hate"),
        ("এদের জ্বালিয়ে দেওয়া উচিত", "hate"),
        ("তোর জাত নিচু", "hate"),
        ("মুসলমানরা সব দেশদ্রোহী", "hate"),
        ("দলিতদের কোনো অধিকার থাকা উচিত না", "hate"),
        ("বিহারিরা আমাদের শহর নোংরা করছে", "hate"),
        ("মেয়েদের কোনো বুদ্ধি নেই", "hate"),
        ("এই ভিখারিগুলো মরে যায় না কেন", "hate"),
        ("তোর মতো কালো লোক কুৎসিত", "hate"),
        ("এরা নর্দমার পোকা", "hate"),
        ("এদের রক্ত ঝরানো উচিত", "hate"),
        ("তোদের সম্প্রদায় সমাজের ক্যান্সার", "hate"),
        ("বিদেশিদের তাড়াও এখান থেকে", "hate"),
        ("সমকামীরা পাগল আর অসুস্থ", "hate"),
        ("তোদের দেশে ফিরে যা", "hate"),
        ("হিন্দুরা সব গোমূত্র পান করে", "hate"),
        ("এই জাতের লোক কোনোদিন উন্নতি করতে পারবে না", "hate"),
        ("মহিলাদের ভোট দেওয়ার অধিকার থাকা উচিত না", "hate"),
        ("এরা সব পরজীবী, কোনো কাজের না", "hate"),

        # === OFFENSIVE ===
        ("তুই বিরাট বোকা রে", "offensive"),
        ("কি বাজে কথা বলছিস", "offensive"),
        ("চুপ কর গাধা", "offensive"),
        ("তোর চেহারা দেখে হাসি পায়", "offensive"),
        ("কত নিচু মানুষ তুই", "offensive"),
        ("তুই এক নম্বরের ভণ্ড", "offensive"),
        ("পাগল নাকি তুই", "offensive"),
        ("তোর চেয়ে বাজে আর কেউ নেই", "offensive"),
        ("এই গান শুনে কান ফেটে গেল", "offensive"),
        ("এই খাবার খেয়ে মরে যাব", "offensive"),
        ("বস আজ মাথা খেয়ে নিল", "offensive"),
        ("ট্রাফিকের জন্য মাথা ফেটে যাচ্ছে", "offensive"),
        ("এই কোম্পানিতে কাজ করা অত্যাচার", "offensive"),
        ("কি বোকামি এই সিস্টেম", "offensive"),
        ("ওকে কিছু আসে না কাজে", "offensive"),
        ("পরীক্ষাটা আমাকে মেরে ফেলছে", "offensive"),
        ("এত খারাপ সিনেমা জীবনে দেখিনি", "offensive"),
        ("তোর রান্না খেয়ে পেট খারাপ হল", "offensive"),
        ("এত বোরিং ক্লাস, ঘুম পাচ্ছে", "offensive"),
        ("এ কি বাজে ডিজাইন, চোখ জ্বলছে", "offensive"),

        # === NEUTRAL ===
        ("আজকের দিনটা খুব সুন্দর", "neutral"),
        ("পরিবারের সাথে সময় কাটানো ভালো লাগে", "neutral"),
        ("সবাইকে ভালোবাসা আর সম্মান দাও", "neutral"),
        ("জীবন খুব সুন্দর", "neutral"),
        ("শিক্ষা সবার অধিকার", "neutral"),
        ("একসাথে কাজ করলে সব সম্ভব", "neutral"),
        ("আজ রান্না খুব সুস্বাদু হয়েছে", "neutral"),
        ("কাল আবহাওয়া খুব ভালো থাকবে", "neutral"),
        ("নতুন বই পড়ে খুব ভালো লাগল", "neutral"),
        ("সবার সমান সুযোগ পাওয়া উচিত", "neutral"),
        ("আজ সূর্যাস্ত অসাধারণ দেখাচ্ছে", "neutral"),
        ("বাগানে নতুন ফুল ফুটেছে", "neutral"),
        ("ক্রিকেট ম্যাচটা খুব উত্তেজনাপূর্ণ ছিল", "neutral"),
        ("গান শুনলে মন ভালো হয়ে যায়", "neutral"),
        ("চা খেতে খেতে বৃষ্টি দেখা ভালো লাগে", "neutral"),
        ("নতুন কিছু শেখা সবসময় ভালো", "neutral"),
        ("স্বেচ্ছাসেবীরা সমাজের জন্য অনেক কিছু করে", "neutral"),
        ("সকালে হাঁটলে শরীর ভালো থাকে", "neutral"),
        ("আজ টিম খুব ভালো খেলেছে", "neutral"),
        ("উৎসবের আমেজে সবাই আনন্দিত", "neutral"),
    ]

    df = pd.DataFrame(samples, columns=["text", "label"])
    df.to_csv(DATA_DIR / "bengali_dataset.csv", index=False)
    print(f"  → Created {len(df)} Bengali samples")
    return df


def create_telugu_dataset():
    """
    Create Telugu + Romanized Telugu hate speech dataset.
    Custom dataset for missing language support.
    """
    print("[TE] Creating Telugu/Roman Telugu dataset...")

    samples = [
        # === HATE (Telugu script) ===
        ("నీ లాంటి వాళ్లను దేశం నుండి తరిమేయాలి", "hate"),
        ("నీ కులం వాళ్లు అందరూ దొంగలు", "hate"),
        ("వీళ్ళు మనుషులు కాదు, జంతువులు", "hate"),
        ("వీళ్ళందరినీ చంపేయాలి", "hate"),
        ("నీ మతం వాళ్ళు అందరూ ఉగ్రవాదులు", "hate"),
        ("వీళ్ళు సమాజానికి పురుగులు", "hate"),
        ("నీ లాంటి మురికి వాళ్ళకి ఇక్కడ చోటు లేదు", "hate"),
        ("వీళ్ళను తగలబెట్టాలి", "hate"),
        ("నీ కులం నీచం", "hate"),
        ("ముస్లింలు అందరూ దేశద్రోహులు", "hate"),
        ("దళితులకు ఎటువంటి హక్కులు ఉండకూడదు", "hate"),
        ("ఈ బిహారీలు మన నగరాన్ని మురికి చేస్తున్నారు", "hate"),
        ("ఆడవాళ్ళకు బుద్ధి ఉండదు", "hate"),
        ("ఈ భిక్షగాళ్ళు చచ్చిపోతే బాగుండు", "hate"),
        ("నీ లాంటి నల్లవాళ్ళు అసహ్యం", "hate"),
        ("వీళ్ళు మురుగు కాలువ పురుగులు", "hate"),
        ("వీళ్ళ రక్తం చిందించాలి", "hate"),
        ("మీ సమాజం సమాజానికి క్యాన్సర్", "hate"),
        ("విదేశీయులను ఇక్కడ నుండి తరమండి", "hate"),
        ("స్వలింగ సంపర్కులు పిచ్చివాళ్ళు, రోగులు", "hate"),
        ("నీ జాతి వాళ్ళు ఏ పనికీ రారు", "hate"),
        ("వీళ్ళందరినీ నాశనం చేయాలి", "hate"),
        ("నీ మతం వల్లే దేశం పాడవుతోంది", "hate"),
        ("ఈ వలస వాళ్ళు మన ఉద్యోగాలు దొంగిలిస్తున్నారు", "hate"),
        ("మీ సంస్కృతి అనాగరికం", "hate"),

        # HATE (Romanized Telugu)
        ("nee lanti vallanu desham nundi tarimeyali", "hate"),
        ("nee kulam vallu andaru dongalu", "hate"),
        ("veellu manushulu kaadu, janthuvulu", "hate"),
        ("veellandarini champeyaali", "hate"),
        ("nee matham vallu andaru ugravadulu", "hate"),
        ("veellu samajaaniki purugulu", "hate"),
        ("nee lanti muriki vallaki ikkada chotu ledu", "hate"),
        ("veellanu tagalabettaali", "hate"),
        ("nee kulam neecham", "hate"),
        ("muslims andaru deshadrohulu", "hate"),
        ("dalitulaku etuvanti hakkulu undakoodadu", "hate"),
        ("ee bihari vallu mana city ni muriki chestunnaru", "hate"),
        ("aadavallaku buddhi undadu", "hate"),
        ("ee bhiksagallu chachchipothe bagundu", "hate"),
        ("nee lanti nallavallu asahyam", "hate"),
        ("nuvvu pichi vadivi, neeku brathakataniki arham ledu", "hate"),
        ("nee jaathi vallu em paniki raaru", "hate"),
        ("veellandarini naasanam cheyaali", "hate"),
        ("nee matham valle desham paadavutundi", "hate"),
        ("ee valasa vallu mana jobs dongilistunnaru", "hate"),

        # === OFFENSIVE (Telugu script) ===
        ("నువ్వు పెద్ద బుద్ధిలేని వాడివి", "offensive"),
        ("ఏం చెత్త మాట్లాడుతున్నావ్", "offensive"),
        ("నోరు మూసుకో గాడిదా", "offensive"),
        ("నీ మొహం చూస్తే నవ్వు వస్తుంది", "offensive"),
        ("ఎంత దిగజారిన మనిషివి నువ్వు", "offensive"),
        ("నువ్వు ఒకటంబర్ వేషగాడివి", "offensive"),
        ("పిచ్చా నీకు పూర్తిగా", "offensive"),
        ("నీ కంటే చెత్త ఇంకెవరు లేరు", "offensive"),
        ("ఈ పాట విని చెవులు పగిలాయి", "offensive"),
        ("ఈ భోజనం తిని చచ్చిపోతాను", "offensive"),
        ("బాస్ ఈ రోజు బుర్ర తినేశాడు", "offensive"),
        ("ట్రాఫిక్ వల్ల తల పగిలిపోతోంది", "offensive"),
        ("ఈ కంపెనీలో పని చేయడం హింస", "offensive"),
        ("ఏం తెలివి తక్కువ సిస్టం ఇది", "offensive"),
        ("వాడికి ఏమీ రాదు పనిలో", "offensive"),

        # OFFENSIVE (Romanized Telugu)
        ("nuvvu pedda buddhileni vadivi", "offensive"),
        ("em chetta maatladutunnav", "offensive"),
        ("noru moosukov gadidha", "offensive"),
        ("nee moham choosteh navvu vasthundi", "offensive"),
        ("entha digajarina manishivi nuvvu", "offensive"),
        ("nuvvu okatambar veshagadivi", "offensive"),
        ("picchaa neeku poorthiga", "offensive"),
        ("nee kante chetta inkevaru leru", "offensive"),
        ("ee paata vini chevulu pagilaayi", "offensive"),
        ("ee bhojanam tini chachchipotha", "offensive"),
        ("boss ee roju burra tineshaadu", "offensive"),
        ("traffic valla tala pagilipotondi", "offensive"),
        ("ee company lo pani cheyadam himsa", "offensive"),
        ("em telivi takkuva system idi", "offensive"),
        ("vadiki emi raadu panilo", "offensive"),

        # === NEUTRAL (Telugu script) ===
        ("ఈ రోజు చాలా మంచి రోజు", "neutral"),
        ("కుటుంబంతో సమయం గడపడం బాగుంటుంది", "neutral"),
        ("అందరికీ ప్రేమ మరియు గౌరవం ఇవ్వండి", "neutral"),
        ("జీవితం చాలా అందమైనది", "neutral"),
        ("విద్య అందరి హక్కు", "neutral"),
        ("కలిసి పని చేస్తే అన్నీ సాధ్యమే", "neutral"),
        ("ఈ రోజు వంట చాలా రుచిగా అయింది", "neutral"),
        ("రేపు వాతావరణం బాగుంటుంది", "neutral"),
        ("కొత్త పుస్తకం చదివి చాలా బాగుంది", "neutral"),
        ("అందరికీ సమాన అవకాశాలు రావాలి", "neutral"),
        ("ఈ రోజు సూర్యాస్తమయం అద్భుతంగా ఉంది", "neutral"),
        ("తోటలో కొత్త పువ్వులు పూశాయి", "neutral"),
        ("క్రికెట్ మ్యాచ్ చాలా ఉత్కంఠభరితంగా ఉంది", "neutral"),
        ("పాటలు వింటే మనసు ప్రశాంతంగా ఉంటుంది", "neutral"),
        ("టీ తాగుతూ వర్షం చూడటం బాగుంటుంది", "neutral"),

        # NEUTRAL (Romanized Telugu)
        ("ee roju chala manchi roju", "neutral"),
        ("kutumbamtho samayam gadapadam baguntundi", "neutral"),
        ("andariki prema mariyu gauravam ivvandi", "neutral"),
        ("jeevitham chala andamainadi", "neutral"),
        ("vidya andari hakku", "neutral"),
        ("kalisi pani chesthe anni saadhyame", "neutral"),
        ("ee roju vanta chala ruchiga ayindi", "neutral"),
        ("repu vaataavaranam baguntundi", "neutral"),
        ("kotha pustakam chadivi chala bagundi", "neutral"),
        ("andariki samaana avakashalu ravaali", "neutral"),
        ("ee movie chuste chala inspire avtham", "neutral"),
        ("morning walk ki velte health ga untundi", "neutral"),
        ("coding nerchukovadam chala fascinating", "neutral"),
        ("team ee roju chala baaga perform chesindi", "neutral"),
        ("festival shopping ayindi, chala enjoy chesam", "neutral"),

        # === MIXED LANGUAGE (Telugu-English) ===
        ("nee lanti waste fellows ni fire cheyali", "hate"),
        ("you are a disgrace to Telugu culture", "hate"),
        ("mee caste vallu eppudu backward untaru", "hate"),
        ("ee movie flop ayindi, director ki sense ledu", "offensive"),
        ("nee coding skills zero ra, learn something", "offensive"),
        ("what a boring lecture, professor ki teaching raadu", "offensive"),
        ("ee new restaurant food amazing undi", "neutral"),
        ("coding in Python nerchukuntunnanu, very interesting", "neutral"),
        ("weekend lo family tho outing plan", "neutral"),
        ("exam lo kill chestha, full preparation done", "neutral"),
        ("adi great performance, stage meeda rock chesadu", "neutral"),
        ("nee presentation lo data chala impressive undi", "neutral"),
        ("this weather is killing my mood but still good day", "offensive"),
        ("nee lanti ugly people ki confidence ekkada nundi vastadi", "hate"),
        ("nee community ni ekkadiki aina pampinchali", "hate"),
    ]

    df = pd.DataFrame(samples, columns=["text", "label"])
    df.to_csv(DATA_DIR / "telugu_dataset.csv", index=False)
    print(f"  → Created {len(df)} Telugu/Roman Telugu samples")
    return df


def create_context_dataset():
    """
    Create context-aware examples to prevent keyword-based false positives.
    Critical for reducing bias and improving context understanding.
    """
    print("[CTX] Creating context-aware dataset...")

    samples = [
        # === Words with "kill/die/death" in non-hate contexts ===
        ("I will kill it on stage tonight", "neutral"),
        ("This exam is killing me right now", "offensive"),
        ("She absolutely killed that dance performance", "neutral"),
        ("These prices are killing my budget", "neutral"),
        ("He killed it with that presentation", "neutral"),
        ("The suspense is killing me in this movie", "neutral"),
        ("I'm dying to try that new restaurant", "neutral"),
        ("I'm dying of laughter watching this comedy", "neutral"),
        ("The deadline is killing my sleep schedule", "neutral"),
        ("She's killing the game with her skills", "neutral"),
        ("This homework is going to be the death of me", "offensive"),
        ("Die-hard fans showed up to the concert", "neutral"),
        ("The comedian killed the audience with his jokes", "neutral"),
        ("That roller coaster ride was killer fun", "neutral"),
        ("I would kill for a pizza right now", "neutral"),

        # "kill/die/death" in hate contexts
        ("I want to kill you and your family", "hate"),
        ("You should die, nobody wants you here", "hate"),
        ("Death to all of them, wipe them out", "hate"),
        ("Someone should kill these people", "hate"),
        ("They deserve to die for being different", "hate"),

        # === Motivational / Positive with strong words ===
        ("Never give up, fight for your dreams", "neutral"),
        ("Destroy your limits and achieve greatness", "neutral"),
        ("Attack every opportunity with full energy", "neutral"),
        ("Crush your goals this year", "neutral"),
        ("Slay your fears and be brave", "neutral"),
        ("Be a warrior, not a worrier", "neutral"),
        ("Fight through the darkness to find the light", "neutral"),
        ("Kill your ego before it kills your growth", "neutral"),
        ("Burn your doubts and rise from the ashes", "neutral"),
        ("Smash every obstacle in your way", "neutral"),
        ("You are stronger than any battle you face", "neutral"),
        ("Be the beast mode version of yourself", "neutral"),
        ("Explode with positivity in everything you do", "neutral"),
        ("Demolish negativity from your life", "neutral"),
        ("Annihilate your excuses and start working", "neutral"),

        # === Sarcasm and figurative language ===
        ("Oh great, another wonderful Monday morning", "neutral"),
        ("Sure, because that's exactly what I needed today", "neutral"),
        ("Wow, what an absolutely brilliant idea... not", "offensive"),
        ("Thanks for nothing, really appreciate it", "offensive"),
        ("What a productive meeting that was... we accomplished nothing", "offensive"),
        ("Oh fantastic, the printer is jammed again", "neutral"),
        ("Clearly the best decision ever made", "neutral"),

        # === News-style sentences ===
        ("The attack resulted in multiple casualties", "neutral"),
        ("Police arrested the suspect for hate crimes", "neutral"),
        ("The bombing killed dozens of civilians", "neutral"),
        ("Terrorist organizations continue to threaten national security", "neutral"),
        ("War crimes tribunal convicts former leader", "neutral"),
        ("Mass shooting reported at local shopping center", "neutral"),
        ("Racial tensions escalate in the city", "neutral"),
        ("Hate crime statistics show alarming increase", "neutral"),
        ("Government condemns recent acts of violence", "neutral"),
        ("Report reveals widespread discrimination in hiring", "neutral"),

        # === Gaming / Sports language ===
        ("I'm going to destroy you in this game", "neutral"),
        ("Get rekt noob, you can't beat me", "offensive"),
        ("Team absolutely demolished the opposition today", "neutral"),
        ("What a savage play by the quarterback", "neutral"),
        ("He murdered that three-pointer, nothing but net", "neutral"),
        ("Our team got slaughtered in the finals", "neutral"),
        ("That was a killer move in chess", "neutral"),
        ("She annihilated the competition in the race", "neutral"),
        ("What a bloodbath of a match, incredible gameplay", "neutral"),
        ("He executed that play perfectly", "neutral"),

        # === Food / Cooking references ===
        ("This curry is fire, absolutely delicious", "neutral"),
        ("The chef murdered that steak, perfectly cooked", "neutral"),
        ("This spice level is killing my taste buds", "neutral"),
        ("These nachos are to die for", "neutral"),
        ("She slayed the baking competition", "neutral"),

        # === Multilingual context examples ===
        ("ye exam mujhe maar daalegi", "offensive"),
        ("aaj khana itna acha hai ki mar jaunga", "neutral"),
        ("uski performance ne sabka dil jeet liya", "neutral"),
        ("usne stage pe aag laga di", "neutral"),
        ("ye gaana sunke pagal ho gaya, kitna acha hai", "neutral"),
        ("mujhe maar do is project se", "offensive"),
        ("uske pass koi talent nahi hai bilkul", "offensive"),
        ("ye film dekhke dimag kharab ho gaya, bakwas", "offensive"),
        ("sabne milke mujhe pagal bana diya, harassment hai", "offensive"),
        ("coding mein naye concepts seekhna exciting hai", "neutral"),
    ]

    df = pd.DataFrame(samples, columns=["text", "label"])
    df.to_csv(DATA_DIR / "context_dataset.csv", index=False)
    print(f"  → Created {len(df)} context-aware samples")
    return df


def merge_all_datasets():
    """
    Merge all individual datasets into a single unified dataset.
    """
    print("\n[MERGE] Merging all datasets...")

    csv_files = list(DATA_DIR.glob("*_dataset.csv"))
    if not csv_files:
        print("  ✗ No dataset files found. Run individual creation functions first.")
        return None

    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["source"] = f.stem
        all_dfs.append(df)
        print(f"  → Loaded {len(df)} samples from {f.name}")

    merged = pd.concat(all_dfs, ignore_index=True)

    # Normalize labels
    label_map = {
        "hate": "hate",
        "offensive": "offensive",
        "neutral": "neutral",
        "hateful": "hate",
        "abusive": "offensive",
        "normal": "neutral",
        "none": "neutral",
    }
    merged["label"] = merged["label"].str.lower().map(label_map).fillna("neutral")

    # Remove duplicates
    merged = merged.drop_duplicates(subset=["text"], keep="first")

    # Shuffle
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    merged.to_csv(DATA_DIR / "merged_dataset.csv", index=False)

    print(f"\n  ✓ Total merged samples: {len(merged)}")
    print(f"  Distribution:\n{merged['label'].value_counts().to_string()}")
    return merged


if __name__ == "__main__":
    print("=" * 60)
    print("  CivicGuard AI — Dataset Collection")
    print("=" * 60)

    create_english_dataset()
    create_hindi_hinglish_dataset()
    create_bengali_dataset()
    create_telugu_dataset()
    create_context_dataset()
    merge_all_datasets()

    print("\n✓ All datasets created and merged successfully!")
    print(f"  Output directory: {DATA_DIR}")
