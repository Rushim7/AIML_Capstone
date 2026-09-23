# analytics.py

import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "analytics"
CSV_FILE = os.path.join(OUTPUT_DIR, "titanic.csv")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
REPORT_FILE = os.path.join(OUTPUT_DIR, "EDA_report.md")


# ============================================================
# Create output folders
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 70)
    print("1. LOADING TITANIC DATASET")
    print("=" * 70)

    try:
        # First and only attempt to load the raw online dataset
        df = sns.load_dataset("titanic")

        print("Titanic dataset loaded using seaborn.")

        # Immediately save offline fallback
        df.to_csv(CSV_FILE, index=False)

        print(f"Offline fallback saved to: {CSV_FILE}")

    except Exception as e:

        print("Could not load Titanic dataset from seaborn.")
        print("Using previously saved offline fallback.")

        if not os.path.exists(CSV_FILE):
            raise FileNotFoundError(
                "No offline titanic.csv file was found."
            ) from e

        df = pd.read_csv(CSV_FILE)

        print(f"Loaded fallback dataset from: {CSV_FILE}")

    return df


# ============================================================
# 2. PROFILE DATASET
# ============================================================

def profile_dataset(df):

    print("\n" + "=" * 70)
    print("2. DATASET PROFILE")
    print("=" * 70)

    print("\n--- df.info() ---")
    df.info()

    print("\n--- df.describe() ---")
    print(df.describe())

    print("\n--- df.shape ---")
    print(df.shape)

    # Missing-value percentage for every column
    missing_percentage = (
        df.isnull()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    missing_percentage = missing_percentage[
        missing_percentage > 0
    ]

    print("\n--- Missing Values ---")

    if missing_percentage.empty:
        print("No missing values found.")
    else:
        for column, percentage in missing_percentage.items():
            print(f"{column}: {percentage:.2f}% missing")

    return missing_percentage


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

def clean_data(df, missing_percentage):

    print("\n" + "=" * 70)
    print("3. MISSING-VALUE HANDLING")
    print("=" * 70)

    cleaned_df = df.copy()

    decisions = []

    # --------------------------------------------------------
    # Process each column with missing values
    # --------------------------------------------------------

    for column, percentage in missing_percentage.items():

        # Under 5% -> drop rows
        if percentage < 5:

            cleaned_df = cleaned_df.dropna(subset=[column])

            decision = (
                f"{column}: {percentage:.2f}% missing -> "
                f"DROPPED rows containing missing values."
            )

            decisions.append(decision)

        # 5% to 30% -> impute
        elif percentage <= 30:

            # Numeric column -> median
            if pd.api.types.is_numeric_dtype(cleaned_df[column]):

                median_value = cleaned_df[column].median()

                cleaned_df[column] = cleaned_df[column].fillna(
                    median_value
                )

                decision = (
                    f"{column}: {percentage:.2f}% missing -> "
                    f"IMPUTED using median ({median_value:.2f})."
                )

            # Categorical column -> mode
            else:

                mode_value = cleaned_df[column].mode()[0]

                cleaned_df[column] = cleaned_df[column].fillna(
                    mode_value
                )

                decision = (
                    f"{column}: {percentage:.2f}% missing -> "
                    f"IMPUTED using mode ('{mode_value}')."
                )

            decisions.append(decision)

        # More than 30% -> drop column
        else:

            cleaned_df = cleaned_df.drop(columns=[column])

            decision = (
                f"{column}: {percentage:.2f}% missing -> "
                f"COLUMN DROPPED because missingness is too high "
                f"for reliable imputation."
            )

            decisions.append(decision)

    print("\nMissing-value decisions:")

    for decision in decisions:
        print(decision)

    print("\nMissing values after cleaning:")

    remaining_missing = cleaned_df.isnull().sum()

    print(
        remaining_missing[
            remaining_missing > 0
        ]
    )

    print("\nCleaned dataset shape:")
    print(cleaned_df.shape)

    return cleaned_df, decisions


# ============================================================
# 4. UNIVARIATE ANALYSIS
# ============================================================

def univariate_analysis(df):

    print("\n" + "=" * 70)
    print("4. UNIVARIATE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # AGE HISTOGRAM
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["age"],
        bins=20,
        kde=True
    )

    plt.title("Distribution of Age")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()

    age_histogram = os.path.join(
        CHART_DIR,
        "age_histogram.png"
    )

    plt.savefig(age_histogram)
    plt.close()

    # --------------------------------------------------------
    # AGE BOX PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x=df["age"]
    )

    plt.title("Age Box Plot")
    plt.xlabel("Age")
    plt.tight_layout()

    age_boxplot = os.path.join(
        CHART_DIR,
        "age_boxplot.png"
    )

    plt.savefig(age_boxplot)
    plt.close()

    # --------------------------------------------------------
    # FARE HISTOGRAM
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["fare"],
        bins=30,
        kde=True
    )

    plt.title("Distribution of Fare")
    plt.xlabel("Fare")
    plt.ylabel("Count")
    plt.tight_layout()

    fare_histogram = os.path.join(
        CHART_DIR,
        "fare_histogram.png"
    )

    plt.savefig(fare_histogram)
    plt.close()

    # --------------------------------------------------------
    # FARE BOX PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x=df["fare"]
    )

    plt.title("Fare Box Plot")
    plt.xlabel("Fare")
    plt.tight_layout()

    fare_boxplot = os.path.join(
        CHART_DIR,
        "fare_boxplot.png"
    )

    plt.savefig(fare_boxplot)
    plt.close()

    # --------------------------------------------------------
    # IQR OUTLIER CALCULATION
    # --------------------------------------------------------

    def calculate_outliers(series):

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]

        return (
            q1,
            q3,
            iqr,
            lower_bound,
            upper_bound,
            len(outliers)
        )

    age_stats = calculate_outliers(df["age"])
    fare_stats = calculate_outliers(df["fare"])

    print("\n--- AGE IQR ---")
    print(f"Q1: {age_stats[0]:.2f}")
    print(f"Q3: {age_stats[1]:.2f}")
    print(f"IQR: {age_stats[2]:.2f}")
    print(f"Lower bound: {age_stats[3]:.2f}")
    print(f"Upper bound: {age_stats[4]:.2f}")
    print(f"Number of outliers: {age_stats[5]}")

    print("\n--- FARE IQR ---")
    print(f"Q1: {fare_stats[0]:.2f}")
    print(f"Q3: {fare_stats[1]:.2f}")
    print(f"IQR: {fare_stats[2]:.2f}")
    print(f"Lower bound: {fare_stats[3]:.2f}")
    print(f"Upper bound: {fare_stats[4]:.2f}")
    print(f"Number of outliers: {fare_stats[5]}")

    # --------------------------------------------------------
    # FARE MEAN / MEDIAN / MODE
    # --------------------------------------------------------

    fare_mean = df["fare"].mean()
    fare_median = df["fare"].median()
    fare_mode = df["fare"].mode().iloc[0]

    print("\n--- FARE STATISTICS ---")
    print(f"Mean: {fare_mean:.2f}")
    print(f"Median: {fare_median:.2f}")
    print(f"Mode: {fare_mode:.2f}")

    # Determine skewness using ordering
    if fare_mean > fare_median > fare_mode:
        skew_description = (
            "Fare is right-skewed because "
            "mean > median > mode."
        )

    elif fare_mean < fare_median < fare_mode:
        skew_description = (
            "Fare is left-skewed because "
            "mean < median < mode."
        )

    else:
        skew_description = (
            "Fare does not follow a simple mean/median/mode "
            "ordering for strong skew classification."
        )

    print("\nFare distribution:")
    print(skew_description)

    return {
        "age_outliers": age_stats[5],
        "fare_outliers": fare_stats[5],
        "fare_mean": fare_mean,
        "fare_median": fare_median,
        "fare_mode": fare_mode,
        "fare_skew": skew_description
    }


# ============================================================
# 5. BIVARIATE ANALYSIS
# ============================================================

def bivariate_analysis(df):

    print("\n" + "=" * 70)
    print("5. BIVARIATE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # SURVIVAL BY SEX
    # --------------------------------------------------------

    sex_results = {}

    for sex in df["sex"].dropna().unique():

        mask = df["sex"] == sex

        survival_rate = df.loc[mask, "survived"].mean()

        sex_results[sex] = survival_rate

    sex_survival = pd.Series(
        sex_results,
        name="survival_rate"
    )

    print("\n--- Survival Rate by Sex ---")
    print(sex_survival)

    # --------------------------------------------------------
    # SURVIVAL BY PCLASS
    # --------------------------------------------------------

    pclass_results = {}

    for pclass in sorted(df["pclass"].dropna().unique()):

        mask = df["pclass"] == pclass

        survival_rate = df.loc[mask, "survived"].mean()

        pclass_results[pclass] = survival_rate

    pclass_survival = pd.Series(
        pclass_results,
        name="survival_rate"
    )

    print("\n--- Survival Rate by Pclass ---")
    print(pclass_survival)

    # --------------------------------------------------------
    # SURVIVAL BY SEX AND PCLASS
    # --------------------------------------------------------

    combined_results = []

    for sex in df["sex"].dropna().unique():

        for pclass in sorted(
            df["pclass"].dropna().unique()
        ):

            mask = (
                (df["sex"] == sex)
                &
                (df["pclass"] == pclass)
            )

            subset = df.loc[mask, "survived"]

            if len(subset) > 0:

                survival_rate = subset.mean()

                combined_results.append({
                    "sex": sex,
                    "pclass": pclass,
                    "survival_rate": survival_rate
                })

    sex_pclass_survival = pd.DataFrame(
        combined_results
    )

    print("\n--- Survival Rate by Sex and Pclass ---")
    print(
        sex_pclass_survival.to_string(index=False)
    )

    # --------------------------------------------------------
    # REQUIRED SIX-COLUMN CORRELATION MATRIX
    # --------------------------------------------------------

    correlation_columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    correlation_matrix = df[
        correlation_columns
    ].corr()

    print("\n--- Required 6x6 Correlation Matrix ---")
    print(correlation_matrix)

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True
    )

    plt.title(
        "Correlation Matrix of Titanic Numeric Features"
    )

    plt.tight_layout()

    correlation_heatmap = os.path.join(
        CHART_DIR,
        "correlation_heatmap.png"
    )

    plt.savefig(correlation_heatmap)
    plt.close()

    # --------------------------------------------------------
    # FIND TWO STRONGEST CORRELATIONS
    # --------------------------------------------------------

    correlation_pairs = []

    for column1, column2 in itertools.combinations(
        correlation_columns,
        2
    ):

        coefficient = correlation_matrix.loc[
            column1,
            column2
        ]

        correlation_pairs.append({
            "feature_1": column1,
            "feature_2": column2,
            "correlation": coefficient,
            "absolute_correlation": abs(coefficient)
        })

    correlation_pairs_df = pd.DataFrame(
        correlation_pairs
    )

    top_two = (
        correlation_pairs_df
        .sort_values(
            "absolute_correlation",
            ascending=False
        )
        .head(2)
    )

    print("\n--- Two Strongest Correlations ---")
    print(
        top_two.to_string(index=False)
    )

    return {
        "sex_survival": sex_survival,
        "pclass_survival": pclass_survival,
        "sex_pclass_survival": sex_pclass_survival,
        "correlation_matrix": correlation_matrix,
        "top_two_correlations": top_two
    }


# ============================================================
# 6. MULTIVARIATE DATA STORY
# ============================================================

def multivariate_analysis(df):

    print("\n" + "=" * 70)
    print("6. MULTIVARIATE DATA STORY")
    print("=" * 70)

    interpretations = {}

    # --------------------------------------------------------
    # CHART 1: Survival by Sex
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df,
        x="sex",
        y="survived"
    )

    plt.title("Survival Rate by Sex")
    plt.xlabel("Sex")
    plt.ylabel("Survival Rate")
    plt.tight_layout()

    chart1 = os.path.join(
        CHART_DIR,
        "survival_by_sex.png"
    )

    plt.savefig(chart1)
    plt.close()

    sex_rates = df.groupby("sex")["survived"].mean()

    interpretations["survival_by_sex"] = (
        f"The survival rate differs substantially by sex. "
        f"The observed survival rates were "
        f"{sex_rates.to_dict()}. "
        f"This indicates that sex was strongly associated with "
        f"survival in this dataset."
    )

    # --------------------------------------------------------
    # CHART 2: Survival by Passenger Class
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df,
        x="pclass",
        y="survived"
    )

    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.tight_layout()

    chart2 = os.path.join(
        CHART_DIR,
        "survival_by_pclass.png"
    )

    plt.savefig(chart2)
    plt.close()

    class_rates = df.groupby("pclass")["survived"].mean()

    interpretations["survival_by_pclass"] = (
        f"Survival rates varied across passenger classes. "
        f"The observed rates for classes 1, 2, and 3 were "
        f"{class_rates.to_dict()}. "
        f"This shows that passenger class was associated with "
        f"different survival outcomes."
    )

    # --------------------------------------------------------
    # CHART 3: Age vs Survival
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="survived",
        y="age"
    )

    plt.title("Age Distribution by Survival")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Age")
    plt.tight_layout()

    chart3 = os.path.join(
        CHART_DIR,
        "age_by_survival.png"
    )

    plt.savefig(chart3)
    plt.close()

    interpretations["age_by_survival"] = (
        "The age distributions of survivors and non-survivors "
        "can be compared directly using the box plot. "
        "The medians and spread show whether age differed between "
        "the two survival groups. The plot also highlights unusually "
        "young or old observations within each group."
    )

    # --------------------------------------------------------
    # CHART 4: Fare vs Survival
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="survived",
        y="fare"
    )

    plt.title("Fare Distribution by Survival")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Fare")
    plt.tight_layout()

    chart4 = os.path.join(
        CHART_DIR,
        "fare_by_survival.png"
    )

    plt.savefig(chart4)
    plt.close()

    interpretations["fare_by_survival"] = (
        "The fare distributions differ between passengers who "
        "survived and those who did not. Higher fares are associated "
        "with the passenger classes that generally had greater access "
        "to resources and higher survival rates. The large upper-tail "
        "values also demonstrate the strong right-skew identified "
        "in the univariate analysis."
    )

    print("\nFour multivariate charts created.")

    return interpretations


# ============================================================
# 7. EXPLORATORY STANDARDIZATION
# ============================================================

def standardization_check(df):

    print("\n" + "=" * 70)
    print("7. EXPLORATORY STANDARDIZATION")
    print("=" * 70)

    standardized_df = df.copy()

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    age_mean = df["age"].mean()
    age_std = df["age"].std()

    standardized_df["age_z"] = (
        df["age"] - age_mean
    ) / age_std

    # --------------------------------------------------------
    # FARE
    # --------------------------------------------------------

    fare_mean = df["fare"].mean()
    fare_std = df["fare"].std()

    standardized_df["fare_z"] = (
        df["fare"] - fare_mean
    ) / fare_std

    # --------------------------------------------------------
    # BEFORE / AFTER SUMMARY
    # --------------------------------------------------------

    before_after = pd.DataFrame({
        "age_before_mean": [df["age"].mean()],
        "age_before_std": [df["age"].std()],
        "age_after_mean": [standardized_df["age_z"].mean()],
        "age_after_std": [standardized_df["age_z"].std()],
        "fare_before_mean": [df["fare"].mean()],
        "fare_before_std": [df["fare"].std()],
        "fare_after_mean": [standardized_df["fare_z"].mean()],
        "fare_after_std": [standardized_df["fare_z"].std()]
    })

    print("\nBefore / After Standardization:")
    print(
        before_after.to_string(index=False)
    )

    print("\nExpected result:")
    print("Standardized age mean ≈ 0")
    print("Standardized age std ≈ 1")
    print("Standardized fare mean ≈ 0")
    print("Standardized fare std ≈ 1")

    return standardized_df, before_after


# ============================================================
# 8. GENERATE README / EDA REPORT
# ============================================================

def create_report(
    df,
    missing_percentage,
    cleaning_decisions,
    univariate_results,
    bivariate_results,
    interpretations,
    before_after
):

    print("\n" + "=" * 70)
    print("8. CREATING EDA REPORT")
    print("=" * 70)

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("# Titanic EDA Report\n\n")

        # ----------------------------------------------------
        # Dataset profile
        # ----------------------------------------------------

        file.write("## 1. Dataset Profile\n\n")

        file.write(
            f"- Rows: **{df.shape[0]}**\n"
        )

        file.write(
            f"- Columns: **{df.shape[1]}**\n\n"
        )

        file.write("### Missing Values\n\n")

        if missing_percentage.empty:

            file.write(
                "No missing values were present.\n\n"
            )

        else:

            for column, percentage in (
                missing_percentage.items()
            ):

                file.write(
                    f"- **{column}**: "
                    f"{percentage:.2f}% missing\n"
                )

            file.write("\n")

        # ----------------------------------------------------
        # Missing-value decisions
        # ----------------------------------------------------

        file.write(
            "## 2. Missing-Value Handling\n\n"
        )

        for decision in cleaning_decisions:

            file.write(
                f"- {decision}\n"
            )

        file.write("\n")

        file.write(
            "Columns with less than 5% missing values had "
            "the affected rows removed. Columns with 5%–30% "
            "missing values were imputed. Columns with more "
            "than 30% missing values were dropped because "
            "imputation would be unreliable given the high "
            "proportion of missing observations.\n\n"
        )

        # ----------------------------------------------------
        # Univariate analysis
        # ----------------------------------------------------

        file.write(
            "## 3. Univariate Analysis\n\n"
        )

        file.write(
            f"- Age IQR outliers: "
            f"**{univariate_results['age_outliers']}**\n"
        )

        file.write(
            f"- Fare IQR outliers: "
            f"**{univariate_results['fare_outliers']}**\n\n"
        )

        file.write(
            f"- Fare mean: "
            f"**{univariate_results['fare_mean']:.2f}**\n"
        )

        file.write(
            f"- Fare median: "
            f"**{univariate_results['fare_median']:.2f}**\n"
        )

        file.write(
            f"- Fare mode: "
            f"**{univariate_results['fare_mode']:.2f}**\n\n"
        )

        file.write(
            f"{univariate_results['fare_skew']}\n\n"
        )

        file.write(
            "### Charts\n\n"
        )

        file.write(
            "![Age Histogram](charts/age_histogram.png)\n\n"
        )

        file.write(
            "![Age Box Plot](charts/age_boxplot.png)\n\n"
        )

        file.write(
            "![Fare Histogram](charts/fare_histogram.png)\n\n"
        )

        file.write(
            "![Fare Box Plot](charts/fare_boxplot.png)\n\n"
        )

        # ----------------------------------------------------
        # Bivariate analysis
        # ----------------------------------------------------

        file.write(
            "## 4. Bivariate Analysis\n\n"
        )

        file.write(
            "### Survival by Sex\n\n"
        )

        file.write(
            bivariate_results[
                "sex_survival"
            ].to_string()
        )

        file.write("\n\n")

        file.write(
            "### Survival by Passenger Class\n\n"
        )

        file.write(
            bivariate_results[
                "pclass_survival"
            ].to_string()
        )

        file.write("\n\n")

        file.write(
            "### Survival by Sex and Passenger Class\n\n"
        )

        file.write(
            bivariate_results[
                "sex_pclass_survival"
            ].to_string(index=False)
        )

        file.write("\n\n")

        # ----------------------------------------------------
        # Correlation
        # ----------------------------------------------------

        file.write(
            "### Correlation Matrix\n\n"
        )

        file.write(
            bivariate_results[
                "correlation_matrix"
            ].to_string()
        )

        file.write("\n\n")

        file.write(
            "The correlation matrix contains exactly these "
            "six numeric columns: survived, pclass, age, sibsp, "
            "parch, and fare. The boolean columns `adult_male` "
            "and `alone` were excluded because they are derived "
            "flags rather than independent measured features.\n\n"
        )

        file.write(
            "### Two Strongest Correlations\n\n"
        )

        top_two = bivariate_results[
            "top_two_correlations"
        ]

        for _, row in top_two.iterrows():

            file.write(
                f"- **{row['feature_1']} and "
                f"{row['feature_2']}**: "
                f"correlation = "
                f"{row['correlation']:.4f}\n"
            )

        file.write("\n")

        file.write(
            "![Correlation Heatmap]"
            "(charts/correlation_heatmap.png)\n\n"
        )

        # ----------------------------------------------------
        # Multivariate story
        # ----------------------------------------------------

        file.write(
            "## 5. Multivariate Data Story\n\n"
        )

        file.write(
            "### Chart 1: Survival by Sex\n\n"
        )

        file.write(
            "![Survival by Sex]"
            "(charts/survival_by_sex.png)\n\n"
        )

        file.write(
            interpretations[
                "survival_by_sex"
            ]
            + "\n\n"
        )

        file.write(
            "### Chart 2: Survival by Passenger Class\n\n"
        )

        file.write(
            "![Survival by Pclass]"
            "(charts/survival_by_pclass.png)\n\n"
        )

        file.write(
            interpretations[
                "survival_by_pclass"
            ]
            + "\n\n"
        )

        file.write(
            "### Chart 3: Age and Survival\n\n"
        )

        file.write(
            "![Age by Survival]"
            "(charts/age_by_survival.png)\n\n"
        )

        file.write(
            interpretations[
                "age_by_survival"
            ]
            + "\n\n"
        )

        file.write(
            "### Chart 4: Fare and Survival\n\n"
        )

        file.write(
            "![Fare by Survival]"
            "(charts/fare_by_survival.png)\n\n"
        )

        file.write(
            interpretations[
                "fare_by_survival"
            ]
            + "\n\n"
        )

        # ----------------------------------------------------
        # Standardization
        # ----------------------------------------------------

        file.write(
            "## 6. Exploratory Standardization\n\n"
        )

        file.write(
            "Age and fare were standardized using the z-score "
            "formula:\n\n"
        )

        file.write(
            "`z = (x - mean) / std`\n\n"
        )

        file.write(
            "### Before / After Summary\n\n"
        )

        file.write(
            before_after.to_string(index=False)
        )

        file.write("\n\n")

        file.write(
            "The standardized columns have means approximately "
            "equal to 0 and standard deviations approximately "
            "equal to 1. This standardization is used only as "
            "an EDA sanity check and is NOT used as input to "
            "the modeling pipeline.\n"
        )

    print(
        f"EDA report saved to: {REPORT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Task 1: Load and save fallback
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Profile original loaded dataset
    # --------------------------------------------------------

    missing_percentage = profile_dataset(df)

    # --------------------------------------------------------
    # Task 2: Clean missing values
    # --------------------------------------------------------

    cleaned_df, cleaning_decisions = clean_data(
        df,
        missing_percentage
    )

    # --------------------------------------------------------
    # Task 3: Univariate analysis
    # --------------------------------------------------------

    univariate_results = univariate_analysis(
        cleaned_df
    )

    # --------------------------------------------------------
    # Task 4: Bivariate analysis
    # --------------------------------------------------------

    bivariate_results = bivariate_analysis(
        cleaned_df
    )

    # --------------------------------------------------------
    # Task 5: Multivariate data story
    # --------------------------------------------------------

    interpretations = multivariate_analysis(
        cleaned_df
    )

    # --------------------------------------------------------
    # Task 6: Exploratory standardization
    # --------------------------------------------------------

    standardized_df, before_after = (
        standardization_check(cleaned_df)
    )

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    create_report(
        cleaned_df,
        missing_percentage,
        cleaning_decisions,
        univariate_results,
        bivariate_results,
        interpretations,
        before_after
    )

    print("\n" + "=" * 70)
    print("ANALYTICS PIPELINE COMPLETE")
    print("=" * 70)

    print(f"\nOffline dataset: {CSV_FILE}")
    print(f"Charts: {CHART_DIR}")
    print(f"Report: {REPORT_FILE}")


# ============================================================
# Run program
# ============================================================

if __name__ == "__main__":
    main()