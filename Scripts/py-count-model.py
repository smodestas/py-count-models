#### Bibliotekų užkrovimas #####

import os
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

#### Darbinės aplinkos nuskaitymas ###

cwd = os.getcwd()

#### Grafikų aplanko sukūrimas #####

try:
    os.mkdir("plots")
except OSError:
    print("Toks aplankas jau egzistuoja")
else:
    print("Aplankas sukurtas")

#### Gražesnio atspausdinimo funkcija ####

def nice_output(klausimas,objektas):
    print(klausimas)
    print(objektas)
    print("-"*50)

#### Duomenų užkrovimas #####

df = pd.read_csv(cwd+"/marketing_campaign.csv", sep="\t")

nice_output("Kaip atrodo duomenys?", df)
nice_output("Kokie kintamieji?", list(df.columns))

### raw duomenis išsaugom, kaip backup ###

df_copy = df

#### Relevantiški stulpeliai analizei #####

df = df[["ID", "Year_Birth", "Education", "Marital_Status", "Income", "MntWines", "Kidhome", "Teenhome"]]

#### Ar duomenys agreguoti? ####

nice_output("Ar duomenys agreguoti vartotojo lygmeniu?", True if len(np.unique(df.ID)) == len(df_copy) else False)

#### Duomenų sutvarkymas #####

### Išmetam ID stulpelį ####

df.drop(columns = "ID", inplace=True)

df["Kids"] = df["Kidhome"] + df["Teenhome"]

df.drop(columns = ["Kidhome", "Teenhome"], inplace=True)

df["Marital_Status"] = df["Marital_Status"].replace({"Married": "Non-single",
                                                     "Together": "Non-single",
                                                     "Single": "Single",
                                                     "Divorced": "Single",
                                                     "Widow": "Single",
                                                     "Alone": "Single",
                                                     "Absurd": "Single",
                                                     "YOLO": "Single"})


#### Ar yra trūkstamų reikšmių ? #####
nice_output("Ar yra trūkstamų reikšmių ?", df[df.isna().any(axis=1)])

#### kadangi NA yra nedaug, tai išmetam. Galima pagalvot apie imputed reiškmes (pvz. prilygint vidurkiui), bet tikriausiai ne šito projekto rėmuose ####
df.dropna(inplace=True)

##### Paskaičiuojam klientų amžių ####

df["Year_Birth"] = 2021 - df.Year_Birth ### dataset aprašyme nėra nurodyta, kurie metai, todėl skaičiuoju nuo šių metų.

df.rename(columns={"Year_Birth": "Age"}, inplace=True)


nice_output("Kaip atrodo sutvarkyti duomenys?", df.head())

#### Duomenų apžvelgimas ####

### Išsilavinimas ####
education = df.Education.value_counts().reset_index().rename(columns={"index": "education",
                                                                      "Education": "freq"})

nice_output("Kokį išsilavinimą turi klientai?", education)

#### Šeimyninė padėtis #####

marital_status = df.Marital_Status.value_counts().reset_index().rename(columns={"index": "marital_status",
                                                                              "Marital_Status": "freq"})

nice_output("Koks klientų šeimyninis statusas?", marital_status)

#### Klientų pajamos ####

nice_output("Kaip atrodo klientų pajamos?", df['Income'].describe())

df = df[df["Income"] != 666666]

nice_output("Klientų pajamos be 666666", df['Income'].describe())

#### Amžius ###

nice_output("Klientų amžius", df['Age'].describe())

### Kiek vaikų turi? ###

nice_output("Kiek vaikų turi?", df['Kids'].describe())

### Pavaizduojam viską grafiškai ####

fig, axes = plt.subplots(2, 3, figsize=(18,10))
fig.suptitle("Customer data")

edu_plot = sns.barplot(ax = axes[0,0],
                        x = "education",
                        y = "freq",
                        data = education)
edu_plot.set(xlabel = '', ylabel = '')
axes[0,0].set_title("Customer education")

marital_plot = sns.barplot(ax = axes[0,1],
                            x = "marital_status",
                            y = "freq",
                            data = marital_status)
marital_plot.set(xlabel = '', ylabel = '')
axes[0,1].set_title("Customer marital status")

kids_plot = sns.barplot(ax = axes[0,2],
                         x = df.Kids.value_counts().reset_index().index,
                         y = df.Kids.value_counts().reset_index().Kids,
                         color = "blue")

kids_plot.set(xlabel = '', ylabel = '')
axes[0,2].set_title("How many kids customers have?")

income_plot = sns.histplot(ax = axes[1,0],
                            x = df.Income)
income_plot.set(xlabel = '', ylabel = '')
income_plot.axvline(np.median(df.Income), color='red',
                 ls='--',
                 lw=2.5)
axes[1,0].set_title("Customer income")

age_plot = sns.histplot(ax = axes[1,1],
                         x = df.Age)
age_plot.axvline(np.mean(df.Age), color='red',
                 ls='--',
                 lw=2.5)
age_plot.set(xlabel = '', ylabel = '')
axes[1,1].set_title("Customer age")

wine_plot = sns.histplot(ax = axes[1,2],
                          x = df.MntWines)
wine_plot.axvline(np.mean(df.MntWines), color='red',
                 ls='--',
                 lw=2.5)
wine_plot.set(xlabel = '', ylabel = '')

axes[1,2].set_title("How much wine customers buy?")

plt.savefig(cwd +"/plots/"+"customer_data.png")

#### Negative binomial regression ####

#### duomenų paruošimas regresijai ####

df = pd.get_dummies(df).drop(columns=["Education_2n Cycle", "Marital_Status_Non-single"])


# Modelio aprašymas
model = """MntWines ~ Age + Income + Kids + Education_Basic + Education_Graduation + Education_Master + Education_PhD 
+ Marital_Status_Single """


negative_binomial_reg = smf.glm(formula = model, data=df, family=sm.families.NegativeBinomial()).fit()

nice_output("Regresijos rezultatai:", negative_binomial_reg.summary())

#### Regresijos rezultatus sunku interpretuoti tiesiogiai, todėl reikia simuliacijų ###

def simuliacija(n_sim, #### simuliacijų skaičius
                Age = None,
                Income = None,
                Kids = None,
                Education = "2nd Cycle",
                Marital_status = "Non-single"):

    # 1. Daugianario skirstinio sukūrimas, naudojant regresijos koeficientus

    mu = np.array(negative_binomial_reg.params.reset_index().iloc[:, 1])
    varcov = np.array(negative_binomial_reg.cov_params())

    S = np.random.multivariate_normal(mu, cov=varcov, size=n_sim)

    # 2. Scenarijaus aprašymas
    scenario = np.array([1, ### Intercept
                         np.mean(df.Age), ### Age
                         np.mean(df.Income), ### Income
                         np.median(df.Kids), ### Number of kids
                         0, ### Basic education
                         0, ### Graduation
                         0, ### Master
                         0, ### PhD
                         0 ###  Marital status
                         ])

    if Age != None:
        scenario[1] = Age

    if Income != None:
        scenario[2] = Income

    if Kids != None:
        scenario[3] = Kids

    if Education == "Basic":
        scenario[4] = 1

    if Education == "Graduation":
        scenario[5] = 1

    if Education == "Master":
        scenario[6] = 1

    if Education == "PhD":
        scenario[7] = 1

    if Marital_status == "Single":
        scenario[8] = 1

    # print(scenario)

    #3. Rezultatų prognozavimas

    lambda1 = np.exp(np.matmul(S, np.transpose(scenario)))

    mean_wine_cons = []

    for i in range(len(lambda1)):
        mean_wine_cons.append(
            np.mean(np.random.negative_binomial(
                negative_binomial_reg.scale, negative_binomial_reg.scale/(negative_binomial_reg.scale + lambda1[i]), 1000)))


    output = [round(np.quantile(mean_wine_cons,0.025),0),
              round(np.quantile(mean_wine_cons,0.5),0),
              round(np.quantile(mean_wine_cons,0.975),0)]
    return output


nice_output("Kiek vidutiniškai vyno butelių nusipirko PhD turintys klientai?",
            simuliacija(1000, Age=47, Education="PhD"))

nice_output("Kiek vidutiniškai vyno butelių nusipirko pagrindinį išsilavinimą turintys klientai?",
            simuliacija(1000, Age=47, Education="Basic"))

####### Pritaikom simuliacijas pajamų range ######

income_range = list(
        range(
            int(np.percentile(df.Income, 25)),
            int(np.percentile(df.Income, 75)),
            1000
        )
    )
results = []

for i in income_range:

    results.append(simuliacija(1000,Income= i, Education="Graduation"))

results = pd.DataFrame(results, columns=["qi_low", "average", "qi_high"])
results.index = income_range
results = results.reset_index().rename(columns={"index": "income"})

#### Pavaizduojam grafiškai ####

fig, ax = plt.subplots(figsize=[15, 10])
ax.plot(results.income,results.average)
ax.fill_between(results.income, results.qi_low, results.qi_high, color='b', alpha=.1)
plt.xlabel('Income')
plt.ylabel('Bottles of Wine bought')
plt.title("How many bottles of wine customers bought based on their income?")
fig.savefig(cwd +"/plots/income_wine_cons.png")

#### Kaip viskas atrodo kintant išsilavinimui? #####

income_range = list(
        range(
            int(np.percentile(df.Income, 25)),
            int(np.percentile(df.Income, 75)),
            1000
        )
    )

results_basic = []

for i in income_range:

    results_basic.append(simuliacija(1000, Income= i, Education="Graduation"))


results_basic = pd.DataFrame(results_basic, columns=["qi_low","average","qi_high"])

results_basic.index = income_range

results_basic = results_basic.reset_index().rename(columns={"index": "income"})

results_basic["Education"] = "Graduation"

results_phd = []

for i in income_range:

    results_phd.append(simuliacija(1000,Income= i, Education="PhD"))


results_phd = pd.DataFrame(results_phd, columns=["qi_low","average","qi_high"])

results_phd.index = income_range

results_phd = results_phd.reset_index().rename(columns={"index": "income"})

results_phd["Education"] = "PhD"

results = results_basic.append(results_phd, ignore_index=True)

#### Pavaizduojam išsialvinimo įtaką vyno pirkimui grafiškai ####

fig, ax = plt.subplots(figsize=[15,10])

ax.plot("income",
        "average",
        data = results[results["Education"] == "Graduation"], label = "Graduation")
ax.fill_between("income", "qi_low", "qi_high", color='b', alpha=.1,
                data=results[results["Education"] == "Graduation"])
ax.plot("income",
        "average",
        data = results[results["Education"] == "PhD"], label = "PhD", color = "r")
ax.fill_between("income", "qi_low", "qi_high", color='r', alpha=.1,
                data=results[results["Education"] == "PhD"])
plt.legend(loc="upper left")
plt.xlabel('Income')
plt.ylabel('Bottles of Wine bought')
plt.title("How many bottles of wine customers bought based on their income and education?")

fig.savefig(cwd +"/plots/edu_wine_cons.png")

plt.show()

