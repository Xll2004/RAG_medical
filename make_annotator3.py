import pandas as pd

TABLE = """
1 | 1 | 1 | 1 | "白睛"→"white" wrong; "五脏"→"blood"; grammar awkward; question lost.
2 | 1 | 1 | 1 | "降糖药"→"degenerative"; "格列吡嗪"→"glycerine"; options have wrong terms; meaning off.
3 | 1 | 1 | 1 | Many terms wrong ("子宫内膜"→"uterine membrane", "宫腔"→"palace cavity"); grammar poor.
4 | 1 | 0 | 1 | "半影"→"half-witted"; question changed to statement; options all wrong.
5 | 1 | 0 | 1 | TCM terms ("阴阳互根") mistranslated; options nonsensical.
6 | 1 | 0 | 1 | "二氧化碳潴留" omitted; "酸中毒"→"Acid poisoning" instead of acidosis; "脑血流"→"Hemorrhage".
7 | 0 | 1 | 1 | Numerical ranges wrongly translated ("一倍"→"half"); options ungrammatical.
8 | 1 | 0 | 1 | "抑菌药"→"incorrigible drug"; "杀菌药"→"Microbicides" (should be bactericidal).
9 | 1 | 0 | 1 | "直立性低血压"→"vertical low blood pressure"; "脉压"→"pulse" missing "pressure".
10 | 1 | 0 | 1 | "鞘膜积液"→"membrane fluid"; "腹股沟内环"→"around the groin"; terms wrong.
11 | 1 | 0 | 1 | "醋酸地塞米松"→"acetic acetic sermons"; "柯柏反应"→"Cobb reacts like a color".
12 | 1 | 1 | 1 | "停经"→"suspended"; "少量出血"→"haemorrhage ... abdominal pains"; grammar broken.
13 | 1 | 0 | 1 | "助阳益阴"→"obsessiveness of the sun"; all herbal names wrong.
14 | 1 | 0 | 1 | "溶血反应"→"soluble blood reactions"; "异丙嗪"→"isopropyl".
15 | 1 | 0 | 1 | "胰岛素瘤"→"insulin tumors" (should be insulinoma); "效果不佳"→"negative results".
16 | 1 | 0 | 1 | "咳嗽变异性哮喘"→"coughing mutant asthma"; "支气管扩张剂"→"bronchial sprawl".
17 | 1 | 0 | 1 | "伪像"→"hypothermia" (completely wrong); "旁瓣效应"→"side valve".
18 | 0 | 0 | 0 | Translation is accurate; no significant errors.
19 | 0 | 0 | 0 | Accurate translation; "层面选择"→"Level Selection" is acceptable.
20 | 1 | 1 | 1 | All drug names wrong; grammar garbled; "醛固酮"→"ellipsine steroids".
21 | 1 | 0 | 1 | "川芎"→"Kawachi"; "抗脑缺血"→"Hemorrhage against the brain"; "解痉"→"Undo it".
22 | 1 | 1 | 1 | "萎缩性鼻炎"→"atrophy of nasitis"; options nonsense (e.g., "innards").
23 | 1 | 0 | 1 | Clinical context omitted; "会阴清洗"→"purge" (should be perineal cleaning).
24 | 1 | 0 | 1 | "血管活性药物"→"angioplastic drugs"; "心肌收缩力"→"constriction" (instead of contractility).
25 | 1 | 0 | 1 | "角膜病"→"Diagonal disease"; "青光眼"→"Cyan-Eye"; "视神经病"→"Psychiatry".
26 | 0 | 0 | 1 | "3-6"→"3-6 p.m." (wrong unit) – semantic drift, but no medical term error.
27 | 1 | 0 | 1 | "新生儿生理性黄疸"→"natural degeneration of a newborn's physiology"; days→"p.m./September".
28 | 1 | 0 | 1 | "不良反应"→"negative response" (should be adverse reaction); meaning reversed.
29 | 1 | 0 | 1 | Clinical details omitted; "输卵管妊娠破裂"→"tubal gestation"; "卵巢肿瘤"→"Neoplasm of ovarians".
30 | 1 | 1 | 1 | "小蓟饮子"→"What's the medicine for a little tasting?"; all options wrong.
31 | 0 | 0 | 1 | "除外"→"Except" but meaning of question slightly off; "经常"→"used to".
32 | 1 | 0 | 1 | Clinical death criteria omitted; "各种反射"→"reflections"; "尸斑"→"body".
33 | 1 | 1 | 1 | "阳明热证/实证"→"positive and positive"; options all mistranslated.
34 | 1 | 0 | 1 | "醋炙"→"When you're jealous" (wrong); rest okay.
35 | 1 | 1 | 1 | "阵发性室性心动过速"→"impregnable room velocity"; grammar broken; drug names wrong.
36 | 1 | 0 | 1 | Many lab terms wrong ("血肌酐"→"blood muscle anhydride", "颗粒管型"→"particle tube type").
37 | 1 | 0 | 1 | "X线照片运动模糊"→"X-line photographs movement obscure"; "心脏搏动"→"heart attack".
38 | 1 | 0 | 0 | "硫代乙酰胺"→"thiomeric acetylamine" (minor terminology error).
39 | 1 | 0 | 0 | "右侧气胸"→"Right air chest", "大叶性肺炎"→"Leaf pneumonia", etc.
40 | 1 | 1 | 1 | "混睛障"→"one-and-the-mortality test"; all TCM eye terms wrong; grammar nonsense.
41 | 1 | 0 | 1 | "杭白芷"→"Hangbeth"; "形成层"→"top form" (should be cambium).
42 | 1 | 0 | 0 | "脂质体"→"Lithium"; "β-环糊精包合物"→"Beta-ring confectionery".
43 | 1 | 0 | 0 | "母病及子"→"Mother and child"; "肝病及脾"→"Hepatitis and spleen" (not exact).
44 | 0 | 0 | 0 | Accurate translation; no errors.
45 | 1 | 0 | 1 | "肺不张" omitted; "后前位"→"back of chest"; "横膈"→"Move horizontally up".
46 | 1 | 0 | 0 | "乳磨牙"→"mammothing"; "窝沟封闭"→"denture closure".
47 | 1 | 0 | 0 | "滴虫阴道炎"→"trichocercia"; "念珠菌"→"Pulse"; "老年性"→"Old-age".
48 | 1 | 0 | 0 | "面部表情疼痛测量图"→"Diagram of facial pain" (should be Faces Pain Scale).
49 | 1 | 1 | 1 | "《金匮要略》"→"Goldblind"; all flavor terms wrong; grammar broken.
50 | 1 | 0 | 0 | "肺气肿"→"pneumoconiosis" (wrong term).
51 | 1 | 0 | 1 | "集中趋势"→"Trends in concentration"; "离散趋势"→"Diffusion trends".
52 | 1 | 0 | 0 | "恶性贫血"→"malignant anaemia" (should be pernicious anaemia).
53 | 1 | 0 | 0 | "糖酵解"→"gluten fermentation"; "乳酸"→"Emulsifiable acid"; "丙酮酸"→"Acetone Acid".
54 | 1 | 0 | 0 | "水飞法"→"hydro-flying"; "赭石"→"A stone of stone"; "朱砂"→"Chu-sa".
55 | 0 | 0 | 0 | Accurate translation; no errors.
56 | 1 | 0 | 1 | "肾精"→"kidney ... glut"; "天癸"→"match-for-fist"; "肝气疏泄"→"Hepaste out".
57 | 1 | 0 | 1 | "小量不保留灌肠"→"not keep enemas" (wrong meaning).
58 | 0 | 0 | 0 | Accurate translation; technical terms correctly rendered.
59 | 1 | 0 | 0 | "甲状腺功能亢进"→"Goiter's inflammation" (should be hyperthyroidism).
60 | 1 | 0 | 0 | Many vitamin B2 properties mistranslated ("硫色素"→"Sulphur colours", "黄素单核苷酸"→"safrole mononucleic acid").
"""

MARK = "√"

rows = {}
for line in TABLE.strip().split("\n"):
    parts = [p.strip() for p in line.split("|", 4)]
    item = int(parts[0])
    rows[item] = {
        "T1": parts[1] == "1",
        "T2": parts[2] == "1",
        "T3": parts[3] == "1",
        "notes": parts[4] if len(parts) > 4 else "",
    }

df = pd.read_excel("annotator1.xlsx")

for col in ["T1_terminology", "T2_syntactic", "T3_semantic", "annotator_notes"]:
    df[col] = None

for idx, r in df.iterrows():
    item = int(r["item_no"])
    if item not in rows:
        continue
    d = rows[item]
    df.at[idx, "T1_terminology"] = MARK if d["T1"] else None
    df.at[idx, "T2_syntactic"] = MARK if d["T2"] else None
    df.at[idx, "T3_semantic"] = MARK if d["T3"] else None
    df.at[idx, "annotator_notes"] = d["notes"]

df.to_excel("annotator3.xlsx", index=False)

print("已写入 annotator3.xlsx")
print("题数:", len(df))
for t, col in [("T1", "T1_terminology"), ("T2", "T2_syntactic"), ("T3", "T3_semantic")]:
    c = df[col].notna().sum()
    print(f"  {t}: {c}/{len(df)} ({c/len(df):.1%})")
print("\n应为 T1 86.7% / T2 20.0% / T3 70.0%")