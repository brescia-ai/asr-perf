# asr-perf
Assessing the performance of ASR systems.

## Datasets

✓ = dataset available for this language
✕ = dataset not available for this language

| Language        | Voxpopuli | MLS | Common Voice 22.0 | Minds14 | Speech-MASSIVE-test | romanian_speech_synthesis_0_8_1 | echo | EuroSpeech |
|-----------------|-----------|-----|-------------------|---------|---------------------|----------------------------------|------|------------|
| Italian (it)    | ✓         | ✓   | ✓                 | ✓       | ✕                   | ✕                                | ✕    | ✓          |
| English (en)    | ✓         | ✕   | ✓                 | ✓       | ✕                   | ✕                                | ✕    | ✓          |
| French (fr)     | ✓         | ✓   | ✓                 | ✓       | ✓                   | ✕                                | ✕    | ✓          |
| German (de)     | ✓         | ✓   | ✓                 | ✓       | ✓                   | ✕                                | ✕    | ✓          |
| Spanish (es)    | ✓         | ✓   | ✓                 | ✓       | ✓                   | ✕                                | ✕    | ✕          |
| Dutch (nl)      | ✓         | ✓   | ✓                 | ✓       | ✓                   | ✕                                | ✕    | ✕          |
| Norwegian (no)  |           |     |                   |         |                     |                                  |      | ✓          |
| Portuguese (pt) | ✕         | ✓   | ✓                 | ✓       | ✓                   | ✕                                | ✕    | ✓          |
| Bulgarian (bg)  | ✕         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Bosnian (bs)    |           |     |                   |         |                     |                                  |      | ✓          |
| Croatian (hr)   | ✓         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Czech (cs)      | ✓         | ✕   | ✓                 | ✓       | ✕                   | ✕                                | ✕    | ✕          |
| Danish (da)     | ✕         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Estonian (et)   | ✓         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Finnish (fi)    | ✓         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Greek (el)      | ✕         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Hungarian (hu)  | ✓         | ✕   | ✓                 | ✕       | ✓                   | ✕                                | ✕    | ✕          |
| Icelandic (is)  |           |     |                   |         |                     |                                  |      | ✓          |
| Latvian (lv)    | ✕         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Lithuanian (lt) | ✓         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Maltese (mt)    | ✕         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Polish (pl)     | ✓         | ✓   | ✓                 | ✓       | ✓                   | ✕                                | ✕    | ✕          |
| Romanian (ro)   | ✓         | ✕   | ✓                 | ✕       | ✕                   | ✓                                | ✓    | ✕          |
| Serbian (sr)    |           |     |                   |         |                     |                                  |      | ✓          |
| Slovak (sk)     | ✓         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Slovenian (sl)  | ✓         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Swedish (sv)    | ✕         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Russian (ru)    | ✕         | ✕   | ✓                 | ✓       | ✓                   | ✕                                | ✕    | ✕          |
| Ukrainian (uk)  | ✕         | ✕   | ✓                 | ✕       | ✕                   | ✕                                | ✕    | ✓          |
| Arabic (ar)     | ✕         | ✕   | ✓                 | ✕       | ✓                   | ✕                                | ✕    | ✕          |
| Korean (ko)     | ✕         | ✕   | ✓                 | ✓       | ✓                   | ✕                                | ✕    | ✕          |
| Turkish (tr)    | ✕         | ✕   | ✓                 | ✓       | ✓                   | ✕                                | ✕    | ✕          |
| Vietnamese (vi) | ✕         | ✕   | ✓                 | ✕       | ✓                   | ✕                                | ✕    | ✕          |

## Results
We consider an AI model ✅ deployable on one given language if its performance is equal or better than the performance of our bs-transcription-1 model on italian.

❓ = insufficient data available for testing

### parakeet-tdt-0.6b-v3
1. ✅ Russian (ru)
2. ✅ German (de)
3. ✅ English (en)
4. ✅ French (fr)
5. ✅ Italian (it)
6. ✅ Spanish (es)
7. ❓ Ukrainian (uk)
8. ✅ Polish (pl)
9. ✅ Romanian (ro)
10. ✅ Dutch (nl)
11. ✅ Czech (cs)
12. ❓ Greek (el)
13. ❓ Swedish (sv)
14. ✅ Portuguese (pt)
15. ✅ Hungarian (hu)
16. Slovak (sk)
17. Finnish (fi)
18. Croatian (hr)
19. Lithuanian (lt)
20. Slovenian (sl)
21. Latvian (lv)
22. Bulgarian (bg)
23. Danish (da)
24. Estonian (et)
25. Maltese (mt)
