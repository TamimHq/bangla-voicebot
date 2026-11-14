# ASR evaluation

Source: `/mnt/user-data/uploads/bangla-voicebot/eval/results/asr_20261114_215342.csv`  
ASR: Gemini (gemini-3.5-flash-lite)  
Scored after normalization (Unicode, punctuation, Bangla digits, accepted spelling variants such as কী/কি). *Wrong-script clips* are Bangla speech written in Latin letters or Hindi script: heard right, written wrong. *Loanword-insensitive* WER also treats common English loanwords as the same word in either script ("balance" = "ব্যালেন্স").

| condition | category | clips | WER | WER, loanword-insensitive | CER | wrong-script clips | WER without them | p50 latency |
|---|---|---|---|---|---|---|---|---|
| clean | **ALL** | 30 | 14.8% | 9.8% | 12.1% | 1 | 12.9% | 1.68s |
| clean | bangla | 15 | 12.6% | 6.3% | 10.8% | 0 | 12.6% | 1.78s |
| clean | code-mixed | 10 | 23.6% | 18.2% | 18.4% | 1 | 18.0% | 1.55s |
| clean | english | 5 | 6.1% | 6.1% | 5.1% | 0 | 6.1% | 1.39s |
| phone | **ALL** | 30 | 14.2% | 5.5% | 13.6% | 0 | 14.2% | 1.38s |
| phone | bangla | 15 | 13.7% | 5.3% | 15.1% | 0 | 13.7% | 1.38s |
| phone | code-mixed | 10 | 21.8% | 7.3% | 17.3% | 0 | 21.8% | 1.55s |
| phone | english | 5 | 3.0% | 3.0% | 1.9% | 0 | 3.0% | 1.33s |

## Worst clips

- [clean] mx05 WER 80% (romanized)
  - ref: recharge করেছি কিন্তু balance আসেনি
  - hyp: recharge kora che kintu balance aheni
- [clean] mx02 WER 60%
  - ref: সবচেয়ে cheap data pack কোনটা
  - hyp: সবচেয়ে ভালো ডেটা প্যাক কোনটা?
- [clean] mx08 WER 60%
  - ref: net speed এত slow কেন
  - hyp: Net speed to slo kno
- [phone] mx02 WER 60%
  - ref: সবচেয়ে cheap data pack কোনটা
  - hyp: সবচেয়ে বেশি ডাটা প্যাক কোনটা?
- [phone] mx08 WER 60%
  - ref: net speed এত slow কেন
  - hyp: নেট স্পিড এত স্লো কেন?
