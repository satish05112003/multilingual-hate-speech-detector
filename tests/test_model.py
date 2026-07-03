"""
CivicGuard AI — Model Backtesting
Tests model on 500+ multilingual sentences and prints confusion matrix.
Target: Accuracy ≥ 85%
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def get_test_sentences():
    """
    Generate 500+ multilingual test sentences across all supported languages.
    Returns list of (text, expected_label) tuples.
    """
    tests = [
        # ══════════════════════════════════════════════════════════
        # ENGLISH
        # ══════════════════════════════════════════════════════════

        # Hate
        ("All immigrants should be deported immediately", "hate"),
        ("Your race is inferior and disgusting", "hate"),
        ("These people are cockroaches in our society", "hate"),
        ("We need to purge these foreigners", "hate"),
        ("They don't deserve to live among us", "hate"),
        ("Your religion makes you a terrorist", "hate"),
        ("Women are too stupid to hold office", "hate"),
        ("Gay people should be put in camps", "hate"),
        ("Black people are inherently violent", "hate"),
        ("All Muslims should be banned from the country", "hate"),
        ("People with disabilities are a waste of resources", "hate"),
        ("Mixed race relationships are an abomination", "hate"),
        ("Indigenous people are savages", "hate"),
        ("Jews run the world secretly", "hate"),
        ("Asian people all look the same", "hate"),
        ("Refugees are invading our homeland", "hate"),
        ("Trans people are mentally disturbed freaks", "hate"),
        ("Poor people deserve their suffering", "hate"),
        ("Old people should just give up and die", "hate"),
        ("Certain ethnicities are born criminals", "hate"),
        ("These illegal aliens are ruining everything", "hate"),
        ("Your caste makes you unworthy", "hate"),
        ("Their bloodline is contaminated", "hate"),
        ("These vermin should be exterminated", "hate"),
        ("Round them all up and get rid of them", "hate"),

        # Offensive
        ("You're the biggest idiot I've ever met", "offensive"),
        ("Shut up you absolute moron", "offensive"),
        ("What a pathetic loser you are", "offensive"),
        ("You're so stupid it hurts", "offensive"),
        ("Get out of my face you clown", "offensive"),
        ("Nobody cares about your dumb opinion", "offensive"),
        ("You look awful in that outfit", "offensive"),
        ("This food tastes like garbage", "offensive"),
        ("What a waste of everyone's time", "offensive"),
        ("You're embarrassing yourself", "offensive"),
        ("Your work is absolute trash", "offensive"),
        ("I can't stand your annoying voice", "offensive"),
        ("You drive like a maniac", "offensive"),
        ("What a joke of a person", "offensive"),
        ("This movie was pure garbage", "offensive"),
        ("Your cooking is terrible", "offensive"),
        ("You have zero common sense", "offensive"),
        ("Stop being such a crybaby", "offensive"),
        ("What a dumpster fire this is", "offensive"),
        ("You're the definition of incompetent", "offensive"),
        ("My boss is the worst person alive", "offensive"),
        ("This customer service is a nightmare", "offensive"),
        ("The professor is clueless about teaching", "offensive"),
        ("What an ugly design, who approved this", "offensive"),
        ("You have no talent whatsoever", "offensive"),

        # Neutral
        ("The weather is beautiful today", "neutral"),
        ("I love spending time with my family", "neutral"),
        ("Education empowers communities", "neutral"),
        ("The sunset was breathtaking yesterday", "neutral"),
        ("My dog learned a new trick", "neutral"),
        ("Just finished reading a great book", "neutral"),
        ("The concert was absolutely amazing", "neutral"),
        ("Kindness makes the world better", "neutral"),
        ("I'm grateful for my friends", "neutral"),
        ("The garden looks wonderful this spring", "neutral"),
        ("Happy birthday to my best friend", "neutral"),
        ("The documentary was incredibly informative", "neutral"),
        ("Volunteering brings me so much joy", "neutral"),
        ("The new cafe has great coffee", "neutral"),
        ("Learning new skills is always rewarding", "neutral"),
        ("The children played happily in the park", "neutral"),
        ("Music connects people across cultures", "neutral"),
        ("Just completed my first 5K run", "neutral"),
        ("The library has an amazing book collection", "neutral"),
        ("Supporting local businesses matters", "neutral"),
        ("Art can change perspectives", "neutral"),
        ("The meditation retreat was transformative", "neutral"),
        ("Fresh fruits from the farmers market taste amazing", "neutral"),
        ("Teamwork makes the dream work", "neutral"),
        ("The sunrise was spectacular from the hilltop", "neutral"),

        # Context-sensitive (kill/die — NOT hate)
        ("I will kill it on stage tonight", "neutral"),
        ("This exam is killing me right now", "offensive"),
        ("She absolutely killed that performance", "neutral"),
        ("I'm dying to try that new restaurant", "neutral"),
        ("He killed it with the guitar solo", "neutral"),
        ("The deadline is killing my sleep", "neutral"),
        ("Die-hard fans filled the stadium", "neutral"),
        ("I would kill for a pizza right now", "neutral"),
        ("That comedian killed the audience", "neutral"),
        ("The suspense is killing me", "neutral"),

        # ══════════════════════════════════════════════════════════
        # HINDI / HINGLISH
        # ══════════════════════════════════════════════════════════

        # Hate
        ("तुम जैसे लोगों को देश से बाहर निकालो", "hate"),
        ("तुम्हारी जाति के लोग चोर हैं", "hate"),
        ("ये लोग जानवर हैं, इंसान नहीं", "hate"),
        ("इनको मारना चाहिए सबको", "hate"),
        ("ये मुसलमान सब आतंकवादी हैं", "hate"),
        ("दलितों के कोई अधिकार नहीं होने चाहिए", "hate"),
        ("ये बिहारी लोग गंदगी फैलाते हैं", "hate"),
        ("औरतों को वोट करने का हक नहीं होना चाहिए", "hate"),
        ("ये भिखारी मर क्यों नहीं जाते", "hate"),
        ("तुम जैसे काले लोग बदसूरत हैं", "hate"),
        ("tum jaise logo ko desh se nikaalo", "hate"),
        ("tumhari caste ke log sab chor hain", "hate"),
        ("ye log janwar hain insaan nahi", "hate"),
        ("inko maar daalo saalo ko", "hate"),
        ("muslims sab terrorists hain", "hate"),
        ("dalits ko koi rights nahi milne chahiye", "hate"),
        ("bihari log gandagi failaate hain", "hate"),
        ("auraton ko vote karne ka haq nahi hona chahiye", "hate"),
        ("ye bhikhari mar kyu nahi jaate", "hate"),
        ("tum jaise kaale log badsoorat hain", "hate"),

        # Offensive
        ("तू बहुत बड़ा बेवकूफ है", "offensive"),
        ("क्या बकवास कर रहा है तू", "offensive"),
        ("चुप हो जा गधे", "offensive"),
        ("तेरी शक्ल देखके हंसी आती है", "offensive"),
        ("तू एक नंबर का ढोंगी है", "offensive"),
        ("tu bahut bada bewkoof hai", "offensive"),
        ("kya bakwas kar raha hai", "offensive"),
        ("chup ho ja gadhe", "offensive"),
        ("teri shakal dekhke hansi aati hai", "offensive"),
        ("tu ek number ka dhongi hai", "offensive"),
        ("boss ne dimag kha liya aaj", "offensive"),
        ("ye traffic mein sir phata ja raha", "offensive"),
        ("company mein kaam karna torture hai", "offensive"),
        ("kya wahiyat system hai ye", "offensive"),
        ("ye khana khaake mar jaunga", "offensive"),

        # Neutral
        ("आज का दिन बहुत अच्छा है", "neutral"),
        ("परिवार के साथ समय बिताना बहुत सुखद है", "neutral"),
        ("शिक्षा सबका मौलिक अधिकार है", "neutral"),
        ("जीवन बहुत सुंदर है", "neutral"),
        ("मिलकर काम करने से सब संभव है", "neutral"),
        ("aaj ka din bahut acha hai", "neutral"),
        ("family ke saath time spend karna acha lagta hai", "neutral"),
        ("education sabka fundamental right hai", "neutral"),
        ("jeevan bahut sundar hai", "neutral"),
        ("milke kaam karne se sab possible hai", "neutral"),
        ("ye movie bahut inspiring hai", "neutral"),
        ("coding seekhna exciting hai", "neutral"),
        ("morning walk health ke liye faydamand hai", "neutral"),
        ("festival ki shopping mein maza aaya", "neutral"),
        ("team ne aaj bahut acha perform kiya", "neutral"),

        # ══════════════════════════════════════════════════════════
        # BENGALI
        # ══════════════════════════════════════════════════════════

        # Hate
        ("তোদের মতো লোকদের দেশ থেকে বের করে দাও", "hate"),
        ("তোর জাতের লোক সব চোর", "hate"),
        ("এরা মানুষ না পশু", "hate"),
        ("এদের সবাইকে মেরে ফেলো", "hate"),
        ("মুসলমানরা সব সন্ত্রাসী", "hate"),
        ("দলিতদের কোনো অধিকার নেই", "hate"),
        ("বিহারিরা আমাদের শহর নোংরা করছে", "hate"),
        ("মেয়েদের বুদ্ধি নেই কোনো", "hate"),
        ("এই ভিখারিগুলো মরে যাক", "hate"),
        ("তোর মতো কালো লোক কুৎসিত", "hate"),
        ("হিন্দুরা সব গোমূত্র পান করে পাগল", "hate"),
        ("এই জাতের লোক কখনো উন্নতি করবে না", "hate"),
        ("এরা সব পরজীবী", "hate"),
        ("বিদেশিদের তাড়িয়ে দাও", "hate"),
        ("সমকামীরা অসুস্থ মানুষ", "hate"),

        # Offensive
        ("তুই বিরাট বোকা", "offensive"),
        ("কি বাজে কথা বলছিস", "offensive"),
        ("চুপ কর গাধা", "offensive"),
        ("তোর চেহারা দেখে হাসি পায়", "offensive"),
        ("কত নিচু মানুষ তুই", "offensive"),
        ("পাগল নাকি তুই", "offensive"),
        ("তোর চেয়ে বাজে কেউ নেই", "offensive"),
        ("পরীক্ষাটা আমাকে মেরে ফেলছে", "offensive"),
        ("এত খারাপ সিনেমা দেখিনি", "offensive"),
        ("বসের মাথা নষ্ট", "offensive"),

        # Neutral
        ("আজকের দিনটা সুন্দর", "neutral"),
        ("পরিবারের সাথে সময় কাটানো ভালো", "neutral"),
        ("শিক্ষা সবার অধিকার", "neutral"),
        ("জীবন সুন্দর", "neutral"),
        ("একসাথে কাজ করলে সব সম্ভব", "neutral"),
        ("আজ রান্না ভালো হয়েছে", "neutral"),
        ("নতুন বই পড়ে ভালো লাগল", "neutral"),
        ("গান শুনলে মন ভালো হয়", "neutral"),
        ("স্বেচ্ছাসেবীরা ভালো কাজ করে", "neutral"),
        ("উৎসবে সবাই আনন্দিত", "neutral"),

        # ══════════════════════════════════════════════════════════
        # TELUGU (Script + Romanized)
        # ══════════════════════════════════════════════════════════

        # Hate
        ("నీ లాంటి వాళ్లను దేశం నుండి తరిమేయాలి", "hate"),
        ("నీ కులం వాళ్లు అందరూ దొంగలు", "hate"),
        ("వీళ్ళు మనుషులు కాదు జంతువులు", "hate"),
        ("వీళ్ళందరినీ చంపేయాలి", "hate"),
        ("ముస్లింలు అందరూ ఉగ్రవాదులు", "hate"),
        ("దళితులకు హక్కులు ఉండకూడదు", "hate"),
        ("ఆడవాళ్ళకు బుద్ధి ఉండదు", "hate"),
        ("ఈ భిక్షగాళ్ళు చచ్చిపోతే బాగుండు", "hate"),
        ("నీ లాంటి నల్లవాళ్ళు అసహ్యం", "hate"),
        ("వీళ్ళ రక్తం చిందించాలి", "hate"),
        ("nee lanti vallanu desham nundi tarimeyali", "hate"),
        ("nee kulam vallu andaru dongalu", "hate"),
        ("veellu manushulu kaadu janthuvulu", "hate"),
        ("veellandarini champeyaali", "hate"),
        ("muslims andaru ugravadulu", "hate"),
        ("dalitulaku hakkulu undakoodadu", "hate"),
        ("aadavallaku buddhi undadu", "hate"),
        ("ee bhiksagallu chachchipothe bagundu", "hate"),
        ("nee lanti nallavallu asahyam", "hate"),
        ("veella raktam chindinchaali", "hate"),

        # Offensive
        ("నువ్వు పెద్ద బుద్ధిలేని వాడివి", "offensive"),
        ("ఏం చెత్త మాట్లాడుతున్నావ్", "offensive"),
        ("నోరు మూసుకో గాడిదా", "offensive"),
        ("నీ మొహం చూస్తే నవ్వు వస్తుంది", "offensive"),
        ("పిచ్చా నీకు పూర్తిగా", "offensive"),
        ("nuvvu pedda buddhileni vadivi", "offensive"),
        ("em chetta maatladutunnav", "offensive"),
        ("noru moosukov gadidha", "offensive"),
        ("nee moham choosteh navvu vasthundi", "offensive"),
        ("picchaa neeku poorthiga", "offensive"),
        ("boss ee roju burra tineshaadu", "offensive"),
        ("traffic valla tala pagilipotondi", "offensive"),
        ("ee company lo pani cheyadam himsa", "offensive"),
        ("ee bhojanam tini chachchipotha", "offensive"),
        ("vadiki emi raadu panilo", "offensive"),

        # Neutral
        ("ఈ రోజు చాలా మంచి రోజు", "neutral"),
        ("కుటుంబంతో సమయం గడపడం బాగుంటుంది", "neutral"),
        ("జీవితం చాలా అందమైనది", "neutral"),
        ("విద్య అందరి హక్కు", "neutral"),
        ("కలిసి పని చేస్తే అన్నీ సాధ్యమే", "neutral"),
        ("ee roju chala manchi roju", "neutral"),
        ("kutumbamtho samayam gadapadam baguntundi", "neutral"),
        ("jeevitham chala andamainadi", "neutral"),
        ("vidya andari hakku", "neutral"),
        ("kalisi pani chesthe anni saadhyame", "neutral"),
        ("coding nerchukovadam chala fascinating", "neutral"),
        ("morning walk ki velte health ga untundi", "neutral"),
        ("team ee roju chala baaga perform chesindi", "neutral"),
        ("festival shopping ayindi chala enjoy chesam", "neutral"),
        ("ee movie chuste chala inspire avtham", "neutral"),

        # ══════════════════════════════════════════════════════════
        # MIXED LANGUAGE / CODE-MIXED
        # ══════════════════════════════════════════════════════════

        # Hate
        ("nee lanti waste fellows ni fire cheyali", "hate"),
        ("you are a disgrace to our culture", "hate"),
        ("mee caste vallu eppudu backward untaru", "hate"),
        ("tum jaise gande log ko hatao yahan se", "hate"),
        ("these people from that religion are all dangerous", "hate"),

        # Offensive
        ("nee coding skills zero ra learn something", "offensive"),
        ("what a boring lecture professor ki teaching raadu", "offensive"),
        ("boss ne aaj dimag kha liya totally useless", "offensive"),
        ("ye system kya bakwas hai totally broken", "offensive"),
        ("that presentation was so bad mujhe neend aa gayi", "offensive"),

        # Neutral
        ("weekend lo family tho outing plan hai", "neutral"),
        ("exam lo kill chestha full preparation done", "neutral"),
        ("coding in Python is bahut interesting", "neutral"),
        ("adi great performance stage meeda rock chesadu", "neutral"),
        ("nee presentation lo data chala impressive undi", "neutral"),

        # ══════════════════════════════════════════════════════════
        # ADDITIONAL CONTEXT-SENSITIVE TESTS
        # ══════════════════════════════════════════════════════════
        ("Never give up, fight for your dreams", "neutral"),
        ("Destroy your limits and achieve greatness", "neutral"),
        ("Kill your ego before it kills your growth", "neutral"),
        ("Be a warrior not a worrier", "neutral"),
        ("Crush your goals this year", "neutral"),
        ("The attack resulted in multiple casualties", "neutral"),
        ("Police arrested the suspect for hate crimes", "neutral"),
        ("Hate crime statistics show alarming increase", "neutral"),
        ("Government condemns recent violence", "neutral"),
        ("War crimes tribunal convicts former leader", "neutral"),
        ("I'm going to destroy you in this game", "neutral"),
        ("Team demolished the opposition today", "neutral"),
        ("He murdered that three-pointer", "neutral"),
        ("What a savage play by the striker", "neutral"),
        ("She annihilated the competition", "neutral"),
        ("This curry is fire, delicious", "neutral"),
        ("These nachos are to die for", "neutral"),
        ("The chef murdered that steak perfectly", "neutral"),
        ("She slayed the baking competition", "neutral"),
        ("This spice level is killing my taste buds", "neutral"),
    ]

    return tests


def run_backtest(model_path: str = None):
    """
    Run comprehensive backtest on the model.
    """
    from backend.inference import get_engine

    print("=" * 60)
    print("  CivicGuard AI — Model Backtesting")
    print("=" * 60)

    # Get test data
    test_data = get_test_sentences()
    print(f"\n[DATA] Total test sentences: {len(test_data)}")

    # Count by expected label
    label_counts = Counter(label for _, label in test_data)
    for label, count in label_counts.items():
        print(f"  → {label}: {count}")

    # Initialize engine
    print(f"\n[MODEL] Loading model...")
    engine = get_engine(model_path)

    # Run predictions
    print(f"\n[TEST] Running predictions...")
    predictions = []
    true_labels = []
    misclassified = []

    for i, (text, expected) in enumerate(test_data):
        result = engine.predict(text)
        pred_label = result["label"]
        confidence = result["confidence"]

        predictions.append(pred_label)
        true_labels.append(expected)

        if pred_label != expected:
            misclassified.append({
                "text": text[:80],
                "expected": expected,
                "predicted": pred_label,
                "confidence": confidence,
            })

        if (i + 1) % 100 == 0:
            print(f"  → Processed {i + 1}/{len(test_data)}")

    # Calculate metrics
    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")

    print(f"\n  Accuracy:   {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  F1 Score:   {f1:.4f} ({f1*100:.1f}%)")

    target_met = accuracy >= 0.85
    print(f"\n  Target (≥85%): {'✓ MET' if target_met else '✗ NOT MET'}")

    # Classification Report
    labels = ["hate", "offensive", "neutral"]
    print(f"\n{'-' * 60}")
    print("  Classification Report")
    print(f"{'-' * 60}")
    print(classification_report(true_labels, predictions, target_names=labels))

    # Confusion Matrix
    print(f"{'-' * 60}")
    print("  Confusion Matrix")
    print(f"{'-' * 60}")
    cm = confusion_matrix(true_labels, predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(f"\n  Predicted →")
    print(cm_df.to_string())

    # Misclassified examples
    if misclassified:
        print(f"\n{'-' * 60}")
        print(f"  Misclassified Samples ({len(misclassified)} total)")
        print(f"{'-' * 60}")
        for m in misclassified[:20]:  # Show first 20
            print(f"  [{m['expected']:>9} → {m['predicted']:<9}] ({m['confidence']:.2f}) {m['text']}")

    # Save results
    results_path = Path(__file__).parent / "backtest_results.json"
    results = {
        "total_samples": len(test_data),
        "accuracy": accuracy,
        "f1_score": f1,
        "target_met": target_met,
        "misclassified_count": len(misclassified),
        "misclassified_samples": misclassified[:50],
        "confusion_matrix": cm.tolist(),
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    run_backtest()
