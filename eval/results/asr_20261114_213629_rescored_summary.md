# ASR evaluation

Source: `/mnt/user-data/uploads/bangla-voicebot/eval/results/asr_20261114_213629.csv`  
ASR: Gemini (gemini-3.5-flash-lite)  
Scored after normalization (Unicode, punctuation, Bangla digits, accepted spelling variants such as কী/কি). *Wrong-script clips* are Bangla speech written in Latin letters or Hindi script: heard right, written wrong. *Loanword-insensitive* WER also treats common English loanwords as the same word in either script ("balance" = "ব্যালেন্স").

| condition | category | clips | WER | WER, loanword-insensitive | CER | wrong-script clips | WER without them | p50 latency |
|---|---|---|---|---|---|---|---|---|
| clean | **ALL** | 30 | 39.9% | 30.1% | 36.1% | 10 | 17.2% | 3.31s |
| clean | bangla | 15 | 48.4% | 36.8% | 44.7% | 6 | 12.5% | 3.84s |
| clean | code-mixed | 10 | 45.5% | 32.7% | 37.5% | 4 | 36.4% | 2.72s |
| clean | english | 5 | 6.1% | 6.1% | 5.1% | 0 | 6.1% | 2.51s |
| phone | **ALL** | 30 | 39.3% | 32.2% | 34.7% | 10 | 15.6% | 2.58s |
| phone | bangla | 15 | 47.4% | 41.1% | 41.6% | 6 | 10.5% | 2.22s |
| phone | code-mixed | 10 | 45.5% | 32.7% | 38.5% | 4 | 34.4% | 2.86s |
| phone | english | 5 | 6.1% | 6.1% | 5.1% | 0 | 6.1% | 5.79s |

## Worst clips

- [phone] bn08 WER 114% (romanized)
  - ref: ফোন চুরি হয়েছে সিমটা বন্ধ করে দিন
  - hyp: Phone-churi hoyeche. SIM-ta bondho kore din.
- [clean] bn03 WER 100% (romanized)
  - ref: এক মাসের জন্য ভালো কোনো প্যাকেজ আছে
  - hyp: ek mader jonno valo kon package ache
- [clean] bn06 WER 100% (romanized)
  - ref: আমার টাকা শেষ ইমার্জেন্সি ব্যালেন্স নিতে চাই
  - hyp: amar taka shesh emergency balance nite chay
- [clean] bn09 WER 100% (romanized)
  - ref: নতুন সিম তুলতে কী কী কাগজ লাগবে
  - hyp: Notun SIM tolte ki ki kagoj lagbe?
- [clean] bn11 WER 100% (romanized)
  - ref: ইন্টারনেট এত ধীরে চলছে কেন
  - hyp: Internet eto dhire cholche keno?
