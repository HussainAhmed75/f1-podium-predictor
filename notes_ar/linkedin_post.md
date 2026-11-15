# LinkedIn post — F1 Podium Predictor

## English (recommended for recruiters)

Monza 2026. Antonelli starts 19th… and wins. 🏁

A model that only looks at the starting grid gave him a 0.06% chance of a podium.
Mine gave him 24%.

Here's how I got there 👇

I took every Formula 1 race ever: 1,169 Grands Prix from Silverstone 1950 to Brazil 2026.
Before building anything, I let the data tell its stories:
🔹 Pole position wins 57% of races in the 2020s, vs 28% in the 1980s.
🔹 Since 2010, starting P1 means an 81% podium chance. Starting P8? Just 3%.
🔹 Cars became almost unbreakable (only 11% failed to finish in 2024), until the new 2026 rules pushed it back to 19%.

Then the real question:
Before the lights go out, what is each driver's chance of finishing on the podium?

The hardest part wasn't the model. It was being honest with the data.
Every feature had to be something you'd know before the race: grid, qualifying gap, team and driver form, championship position.
No peeking into the future. I even wrote a test to prove it.

I trained on 2006–2022, tuned LightGBM with Optuna, and tested on 90 races from 2023–2026 it had never seen:
✅ ROC-AUC 0.943 (grid-only: 0.919)
✅ 14% lower log-loss
✅ Picked the race winner 67.8% of the time (betting on pole: 61.1%)

And the honest part: for naming the 3 podium finishers, it only ties the grid order.
In modern F1, the grid already says a lot. The model's edge is knowing when the grid is lying, like Monza.

Built with Python, scikit-learn, LightGBM, Optuna, SHAP, Plotly and Streamlit, plus an interactive dashboard and a what-if app.

🔗 Code: https://github.com/HussainAhmed75/f1-podium-predictor

If you follow F1: should a model trust the grid, or the team's form? 👇

#MachineLearning #DataScience #Python #F1 #Formula1 #LightGBM #DataAnalysis #Portfolio

---

## بالمصري

مونزا 2026. أنتونيلي بيبدأ السباق من المركز 19… وبيكسبه 🏁

موديل بيبص على الجريد بس كان مدّيله 0.06% يطلع على البوديوم.
الموديل بتاعي كان مدّيله 24%.

الحكاية بدأت إني جبت كل سباقات الفورمولا 1 في التاريخ: 1,169 سباق، من سيلفرستون 1950 لحد البرازيل 2026.
وقبل ما أبني أي موديل، سيبت الداتا تحكي:
🔹 اللي بيبدأ من البول بقى بيكسب 57% من السباقات في العشرينات، وفي التمانينات كانت 28%.
🔹 من 2010، اللي بيبدأ أول بياخد بوديوم 81% من المرات، واللي بيبدأ تامن 3% بس.
🔹 العربيات بقت تقريباً مبتعطلش (11% بس عطلت في 2024)، لحد ما قوانين 2026 رجّعتها 19%.

وبعدين السؤال الحقيقي:
قبل ما السباق يبدأ، إيه احتمال كل سواق يطلع على البوديوم؟

أصعب حاجة مكانتش الموديل. أصعب حاجة كانت إني أبقى أمين مع الداتا.
كل معلومة الموديل بيتعلم منها لازم تكون معروفة قبل السباق: الجريد، والتصفيات، وفورم الفريق والسواق.
ممنوع يبص على المستقبل، وكتبت اختبار مخصوص يثبت كده.

دربت على 2006–2022، وعملت tuning لـ LightGBM بـ Optuna، واختبرت على 90 سباق من 2023 لـ 2026 الموديل عمره ما شافهم:
✅ ROC-AUC 0.943، وموديل الجريد بس 0.919
✅ Log-loss أقل بـ 14%
✅ توقع الفايز صح 67.8% من المرات، واللي يراهن على البول بيصيب 61.1%

والجزء الأمين: في اختيار التلاتة اللي هيطلعوا على البوديوم، الموديل متعادل مع ترتيب الجريد.
الجريد في الفورمولا 1 الحديثة بيقول كتير. قيمة الموديل إنه يعرف إمتى الجريد بيكدب، زي مونزا.

🔗 الكود: https://github.com/HussainAhmed75/f1-podium-predictor

لو بتتابع الفورمولا 1: تصدق الجريد ولا فورم الفريق؟ 👇

#MachineLearning #DataScience #Python #F1 #تعلم_الآلة #تحليل_بيانات
