# P13 Presentation Outline
## "The Aesthetics of Generation: Stylometric Fingerprints of LLMs"

**Duration:** 10 minutes (12 slides, ~50 seconds per slide)  
**Audience:** Prof. Alfio Ferrara + NLP class  
**Format:** PowerPoint / Google Slides

---

## Slide-by-Slide Breakdown

### SLIDE 1: Title Slide
**Text:**
- Title: "The Aesthetics of Generation"
- Subtitle: "Can we identify which LLM wrote this?"
- Author: [Your Name]
- Course: NLP Final Project | Prof. Alfio Ferrara | May 2026

**Talking Points (30 sec):**
- "Today I'm investigating whether large language models have distinct 'voices' or stylistic signatures"
- "Think of it like forensic authorship attribution, but for AI"
- "The question: Can a classifier tell ChatGPT from Claude from Gemini just by reading the text?"

**Visual:** Keep minimal, maybe add a subtle background with three different colored LLM logos

---

### SLIDE 2: The Research Question
**Text:**
```
Do Large Language Models have measurable,
identifiable "writing styles"?

🔍 Can we recognize which model wrote a text
   based on stylometric features alone?

📊 How robust are these signatures across
   different genres and prompts?

🎯 Which features distinguish model "voices"?
```

**Talking Points (40 sec):**
- "This is fundamentally about authorship attribution in AI"
- "Unlike human authors who develop style through experience, LLMs generate text probabilistically"
- "Yet preliminary evidence suggests each model leaves a recognizable fingerprint"
- "Our hypothesis: YES, we can identify models with high accuracy using only stylometric analysis"

**Visual:** Question mark icon, maybe split-screen showing ChatGPT/Gemini/Claude logos

---

### SLIDE 3: Why This Matters
**Text:**
```
🔐 Model Fingerprinting
   Identify AI-generated content in academic/professional contexts

📋 Content Attribution
   Verify which model was used for document generation

🧠 Model Understanding
   Understand how architecture shapes language production

🛡️ AI Safety & Detection
   Develop better detection of synthetic content
```

**Talking Points (30 sec):**
- "As LLMs proliferate, we need reliable methods to identify their output"
- "This has applications in academic integrity, content verification, and AI safety"
- "Understanding *why* models have distinct styles deepens our understanding of language generation itself"

**Visual:** Icons for each point; maybe security/shield imagery

---

### SLIDE 4: Dataset Design
**Text:**
```
📊 CORPUS DESIGN
─────────────────────────────────
Models:          3 (ChatGPT, Gemini, Claude)
Genres:          5 (Narrative, Argumentative, 
                    Descriptive, Dialogue, Creative)
Total Texts:     105 (perfectly balanced)
  Per model:     35 texts
  Per genre:     7 per model

✓ Constant prompts across all models
✓ ~250 words per text
✓ No cherry-picking (first natural response)
```

**Talking Points (40 sec):**
- "We designed a balanced corpus to isolate stylistic variation from content variation"
- "Each model answered the same 35 prompts across 5 different genres"
- "This ensures we're comparing like-to-like"
- "Think of it like giving all three models a writing test with identical questions"

**Visual:** Table showing model × genre breakdown (3×5 grid with 7 in each cell)

---

### SLIDE 5: Feature Engineering
**Text:**
```
🔧 170 STYLOMETRIC FEATURES
──────────────────────────────────
7   Lexical Features
    Type-Token Ratio, MTLD, Hapax, Yule's K, word length...

18  Syntactic Features
    POS ratios, sentence length, dependency depth...

5   Readability Features
    Flesch, Flesch-Kincaid, SMOG, Dale-Chall...

2   Discourse Features
    Discourse markers (however, therefore, etc.)

11  Punctuation Features
    Em-dash, comma, semicolon, quotes...

~150 Function Words (Mosteller-Wallace)
    Articles, prepositions, pronouns...
```

**Talking Points (50 sec):**
- "We extracted 170 interpretable features—not black-box embeddings"
- "This gives us transparency: we know exactly what we're measuring"
- "Features range from simple (word length) to sophisticated (syntactic depth, discourse markers)"
- "Function words are especially powerful for authorship attribution—authors unconsciously repeat patterns"
- "These aren't features we invented; they're standard in computational stylometry literature"

**Visual:** Stack/pyramid showing the 7 families; color-code each family

---

### SLIDE 6: The Big Result
**Text:**
```
🎯 CLASSIFICATION ACCURACY

            97.1% ± 3.8%
            ═════════════════════

vs. Baseline (random guessing)    33.3%

Statistical Significance
    Permutation test p-value = 0.005
    (The signal is REAL, not by chance)

Misclassifications: Only 3/105 texts
```

**Talking Points (60 sec - THIS IS YOUR HEADLINE):**
- "Here's the main result: We achieved 97.1% accuracy in identifying which model wrote a text"
- "That's compared to a 33.3% chance baseline—like randomly guessing between 3 options"
- "We validated this with a permutation test: if we shuffled labels randomly, we'd only get ~33%"
- "The fact that we get 97% means the signal is statistically real, not due to overfitting"
- "Only 3 misclassifications out of 105 texts"
- "This demonstrates that LLMs DO have measurable stylistic signatures"

**Visual:** LARGE bar chart: 97.1% (ChatGPT) vs 33.3% (Chance), maybe with error bars showing ±3.8%

---

### SLIDE 7: Top Discriminative Features
**Text:**
```
🏆 TOP 3 FEATURES DISTINGUISHING MODELS

1️⃣  EM-DASH FREQUENCY (F = 50.7)
    ChatGPT: 2.2× more than Gemini
    → Creates "emphatic, structural" style

2️⃣  ARTICLE "A" FREQUENCY (F = 39.1)
    ChatGPT: Higher  |  Claude: Lower
    → Reveals function-word preferences

3️⃣  AUXILIARY VERB RATIO (F = 29.8)
    ChatGPT: 0.156  |  Gemini: 0.122
    → Reflects syntactic complexity choices
```

**Talking Points (50 sec):**
- "These features emerged from ANOVA testing across all 170 features"
- "The F-statistic tells us how much each feature differs between models"
- "Em-dash is the #1 discriminator: ChatGPT uses em-dashes ~2.2 times more frequently"
- "This creates a more 'emphatic' style with clear rhetorical breaks"
- "Function words (articles, auxiliaries) reveal unconscious stylistic preferences"
- "These are the features our classifier relies on most"

**Visual:** Three cards showing the top features with the F-statistics as large numbers; perhaps show example text snippets for em-dash

---

### SLIDE 8: Robustness Testing
**Text:**
```
✅ WE STRESS-TESTED THE SIGNAL

Leave-One-Genre-Out Cross-Validation
    → 93.3% accuracy (vs. 33% chance)
    ✓ Stylistic signal robust ACROSS genres

Statistical Significance
    → 36/170 features significant (p<0.05, Holm-corrected)
    ✓ Strong feature signal across the board

Effect Size Analysis (Cohen's d)
    → Large effect sizes (d > 0.8) for top features
    ✓ Differences are NOT just statistically real, but PRACTICALLY meaningful

Permutation Feature Importance
    → Top 20 features explain 95% of predictive power
    ✓ Clear feature hierarchy
```

**Talking Points (50 sec):**
- "We didn't just achieve high accuracy on random splits—we rigorously validated robustness"
- "Leave-one-genre-out tests: train on 4 genres, test on the 5th held-out genre"
- "Result: 93.3% accuracy—the signal generalizes across different writing contexts"
- "We also performed ANOVA testing: 36 out of 170 features differ significantly between models"
- "Effect sizes (Cohen's d) show these aren't tiny differences—they're large, practically meaningful"
- "Most importantly: top 20 features account for 95% of predictive power—no feature bloat"

**Visual:** Two or three horizontal bar charts showing robustness metrics; maybe a confusion matrix inset

---

### SLIDE 9: Model Profiles
**Text:**
```
📝 WHAT EACH MODEL'S "VOICE" REVEALS

🤖 ChatGPT
   Emphatic, structured: Heavy em-dash use, formal tone
   Characteristic: "Clear reasoning" style with rhetorical breaks

🔍 Gemini
   Flowing, balanced: Consistent article usage, narrative fluency
   Characteristic: "Natural rhythm" with smooth transitions

🧠 Claude
   Deliberate, explicit: Discourse markers signal reasoning
   Characteristic: "Verbose clarity" with hedging language
```

**Talking Points (40 sec):**
- "Each model has a distinct stylometric personality"
- "ChatGPT's heavy em-dash use creates an emphatic, 'structured reasoning' style"
- "Gemini's consistent article patterns and smoother prose suggest narrative fluency"
- "Claude's preference for discourse markers and careful qualification creates a 'explicit reasoning' style"
- "These aren't just statistical artifacts—they align with known differences in model design"

**Visual:** Three character profiles or personas; maybe show radar charts for each model's top features

---

### SLIDE 10: Visualizations
**Text:**
```
📊 HOW MODELS CLUSTER IN STYLE SPACE

PCA (Principal Component Analysis)
    ↓
    Clear separation of models in 2D space
    Minimal overlap = distinct stylistic fingerprints

t-SNE (t-Distributed Stochastic Neighbor Embedding)
    ↓
    More dramatic clustering (nonlinear projection)
    Models form distinct visual regions

Confusion Matrix
    ↓
    102/105 texts correctly classified
    Only 3 errors (all in confused pairs)
```

**Talking Points (40 sec):**
- "When we visualize the 170-dimensional feature space in 2D using PCA, we see clear model separation"
- "t-SNE creates even more dramatic clustering, showing how distinct each model's 'voice' is"
- "The confusion matrix shows where our classifier struggles: a few borderline texts that share features"
- "But overwhelmingly, texts cluster by model, not by genre or prompt—the style signal dominates"

**Visual:** Show 2-3 key figures:
- `pca.png` (2D PCA scatter with 3 colors)
- `confusion_matrix.png` (3×3 heatmap)
- Maybe embed `confusion_sankey.html` screenshot or description

---

### SLIDE 11: Implications & Limitations
**Text:**
```
💡 WHAT THIS MEANS

✓ LLMs DO have measurable stylistic signatures
✓ These signatures are robust, statistically real, and practically distinguishable
✓ Implications: Model fingerprinting, AI-content detection, style standardization
✓ Contributes to emerging field of "AI stylistics"

⚠️  KNOWN LIMITATIONS

• Corpus size: 105 texts (could expand to 300+)
• Only 3 commercial models (not open-source like LLaMA)
• Single collection date (temporal variation untested)
• Style transfer experiments not yet implemented (optional)
• No human baseline (due to time constraints)
```

**Talking Points (50 sec):**
- "This research demonstrates that LLMs are NOT stylistically neutral"
- "Each model's training procedure, architecture, and alignment create recognizable patterns"
- "This has real-world implications: we can now reliably detect which model generated a text"
- "This matters for content attribution, academic integrity, AI safety"
- "That said, we're honest about limitations: this is a proof-of-concept on 105 texts"
- "Future work: expand to more models, test temporal variation, implement style mimicry experiments"

**Visual:** Left side showing implications (checkmarks, green), right side showing limitations (warnings, orange)

---

### SLIDE 12: Conclusion
**Text:**
```
🎯 MAIN TAKEAWAY

Large Language Models exhibit DISTINCT,
MEASURABLE, and ROBUST stylistic signatures.

These signatures can be reliably identified
with 97.1% accuracy using only stylometric
features, far exceeding chance baseline (33%).

This contributes to the emerging field of
AI stylistics and supports the hypothesis
that LLM "voice" is a real linguistic phenomenon.

───────────────────────────────────────────
Questions?
```

**Talking Points (40 sec - CLOSING REMARKS):**
- "In conclusion: Yes, LLMs have style. We proved it rigorously."
- "97.1% accuracy demonstrates that stylistic signatures are strong, consistent, and identifiable"
- "This work opens questions about what 'creativity' and 'voice' mean in AI-generated text"
- "And it provides practical tools for model fingerprinting and AI-content detection"
- "Thank you—I'm happy to answer questions about the methodology, the data, or the implications"

**Visual:** Simple, clean slide; maybe a summary stat or quote

---

## 🎬 Presentation Tips

### Timing Breakdown:
- Slide 1 (Title): 30 sec
- Slide 2 (Question): 40 sec
- Slide 3 (Why Matter): 30 sec
- Slide 4 (Dataset): 40 sec
- Slide 5 (Features): 50 sec
- **Slide 6 (97.1% Result): 60 sec** ← SPEND TIME HERE, THIS IS YOUR HEADLINE
- Slide 7 (Top Features): 50 sec
- Slide 8 (Robustness): 50 sec
- Slide 9 (Model Profiles): 40 sec
- Slide 10 (Visualizations): 40 sec
- Slide 11 (Implications): 50 sec
- Slide 12 (Conclusion): 40 sec

**Total: ~10-11 minutes** (leaves 1-2 min for Q&A)

### Delivery Notes:
1. **Pace:** Slow down on Slides 6-8 (the technical results)
2. **Emphasis:** Stress that 97.1% >> 33.3% chance
3. **Tone:** Mix technical rigor with accessibility
4. **Narrative:** Question → Why It Matters → Methodology → BIG RESULT → Validation → Implications
5. **Energy:** Peak energy on Slide 6 (your headline result)

### What to Bring:
- This outline as speaker notes
- Backup plan: show `FOR_PROFESSOR.md` if tech fails
- PDF export of key figures from `results/figures/`

---

## 📁 WHERE TO CREATE THE PRESENTATION

### Option 1: PowerPoint (.pptx) — **RECOMMENDED**
**Create in:** `C:\Users\93613828\TEST\p13-stylometry\presentation\` (new folder)  
**Filename:** `P13_Presentation.pptx`

**Steps:**
1. Create folder: `presentation/` in project root
2. Open PowerPoint → Blank presentation
3. Use this outline for each slide
4. Import images from `results/figures/`:
   - Slide 6: `feature_importance.png`
   - Slide 9: `radar_profile.png`
   - Slide 10: `pca.png`, `confusion_matrix.png`
5. Save to `presentation/P13_Presentation.pptx`

### Option 2: Google Slides (Cloud)
**Create:** Google Slides document  
**Share:** Get link to present live  
**Advantages:** No file management, can present from anywhere

### Option 3: Markdown + Marp (If you want to keep it in your repo)
**Create in:** `presentation.md` (in root)  
**Convert to PDF via:** Marp online tool or CLI  
**Filename:** `P13_Presentation.pdf`

---

## 📊 Suggested Slide Images

| Slide # | Image File | Caption |
|---------|-----------|---------|
| 1 | (custom title design) | — |
| 6 | Bar chart (97.1% vs 33.3%) | Accuracy comparison |
| 7 | Feature rankings | Top 3 discriminative features |
| 8 | Cross-validation results | Robustness metrics |
| 9 | `radar_profile.png` | Model stylometric profiles |
| 10 | `pca.png` + `confusion_matrix.png` | Clustering & confusion |
| 11 | Icons/diagrams | Implications vs limitations |

---

## 🎤 Speaker Notes Template

For each slide, I've included ~30-60 second talking points above. Print these as speaker notes:
- Open PowerPoint → View → Notes Page
- Copy the talking points into Notes section for each slide
- Practice reading them naturally (not reading verbatim—use as reminders)

---

## ✅ Presentation Checklist

Before presenting:
- [ ] All 12 slides created
- [ ] Images embedded (pca.png, confusion_matrix.png, radar_profile.png)
- [ ] Speaker notes added to each slide
- [ ] Timing rehearsed (should be 10-11 minutes)
- [ ] Can access `results/figures/` as backup
- [ ] Tested video/audio if presenting remotely
- [ ] PDF backup exported (just in case)

---

**Ready to build? Start with Option 1 (PowerPoint) — it's the easiest!**
