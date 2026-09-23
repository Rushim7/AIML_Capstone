
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"

# Required project-defined conversion rate
GBP_TO_INR = 105.50

# Scrape first 5 catalogue pages
NUMBER_OF_PAGES = 5

# Database location
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "books.db"

# Output file for SQL query results
QUERY_OUTPUT_PATH = BASE_DIR / "query_outputs.txt"


# ============================================================
# PHASE 1: SCRAPING
# ============================================================

def scrape_books():
    """
    Scrape books from the first 5 catalogue pages.

    Returns:
        pandas.DataFrame containing raw scraped data.
    """

    all_books = []

    print("Starting web scraping...")

    for page_number in range(1, NUMBER_OF_PAGES + 1):

        url = BASE_URL.format(page_number)

        print(f"Scraping page {page_number}: {url}")

        response = requests.get(url, timeout=10)

        # Explicitly check HTTP status
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        books = soup.select("article.product_pod")

        for book in books:

            # -----------------------------
            # Title
            # -----------------------------
            title_element = book.select_one("h3 a")

            title = (
                title_element.get("title", "").strip()
                if title_element
                else None
            )

            # -----------------------------
            # Price
            # -----------------------------
            price_element = book.select_one(".price_color")

            price = (
                price_element.get_text(strip=True)
                if price_element
                else None
            )

            # -----------------------------
            # Star rating
            # -----------------------------
            rating_element = book.select_one(".star-rating")

            star_rating = None

            if rating_element:
                classes = rating_element.get("class", [])

                # Classes look like:
                # ["star-rating", "Three"]
                if len(classes) >= 2:
                    star_rating = classes[1]

            # -----------------------------
            # Availability
            # -----------------------------
            availability_element = book.select_one(
                ".availability"
            )

            availability = (
                availability_element.get_text(" ", strip=True)
                if availability_element
                else None
            )

            # -----------------------------
            # Category
            # -----------------------------
            #
            # The category is not available directly on the
            # individual product card.
            #
            # Books to Scrape requires visiting each book's
            # detail page to obtain its category.
            # -----------------------------

            book_link = (
                title_element.get("href")
                if title_element
                else None
            )

            category = get_book_category(
                book_link,
                url
            )

            all_books.append({
                "title": title,
                "price": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": category
            })

    df = pd.DataFrame(all_books)

    print(f"\nTotal books scraped: {len(df)}")

    return df


# ============================================================
# GET CATEGORY FROM BOOK DETAIL PAGE
# ============================================================

def get_book_category(book_link, page_url):
    """
    Visit the book detail page and retrieve its category.
    """

    if not book_link:
        return None

    # Convert relative URL into full URL
    detail_url = requests.compat.urljoin(
        page_url,
        book_link
    )

    try:

        response = requests.get(
            detail_url,
            timeout=10
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Breadcrumb structure:
        #
        # Home > Books > Category > Book Title
        #
        breadcrumb = soup.select(
            "ul.breadcrumb li"
        )

        if len(breadcrumb) >= 3:

            category = breadcrumb[2].get_text(
                strip=True
            )

            return category

    except requests.RequestException as error:

        print(
            f"Could not retrieve category "
            f"for {detail_url}: {error}"
        )

    return None


# ============================================================
# PHASE 2: CLEANING
# ============================================================

def clean_data(df):
    """
    Clean raw scraped fields and create properly typed columns.
    """

    print("\nCleaning data...")

    cleaned = df.copy()

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    # Example:
    # £51.77 -> 51.77

    cleaned["price_gbp"] = (
        cleaned["price"]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.strip()
    )

    cleaned["price_gbp"] = pd.to_numeric(
        cleaned["price_gbp"],
        errors="coerce"
    )

    # Median imputation for numeric field
    price_median = cleaned["price_gbp"].median()

    cleaned["price_gbp"] = cleaned["price_gbp"].fillna(
        price_median
    )

    # --------------------------------------------------------
    # STAR RATING
    # --------------------------------------------------------

    rating_mapping = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    cleaned["rating"] = cleaned["star_rating"].map(
        rating_mapping
    )

    # Rating is numeric, so use median imputation
    rating_median = cleaned["rating"].median()

    cleaned["rating"] = cleaned["rating"].fillna(
        rating_median
    )

    cleaned["rating"] = cleaned["rating"].astype(int)

    # --------------------------------------------------------
    # AVAILABILITY
    # --------------------------------------------------------

    # Books to Scrape normally returns:
    # "In stock (22 available)"
    #
    # If the text contains "In stock", mark True.
    # Otherwise False.

    cleaned["in_stock"] = (
        cleaned["availability"]
        .fillna("")
        .str.contains(
            "In stock",
            case=False,
            na=False
        )
    )

    # Convert explicitly to boolean
    cleaned["in_stock"] = cleaned["in_stock"].astype(bool)

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    # Drop rows where category/title cannot be obtained.
    # These are essential fields for the normalized database.

    cleaned = cleaned.dropna(
        subset=["title", "category"]
    )

    # --------------------------------------------------------
    # PRICE INR
    # --------------------------------------------------------

    # Required project-defined constant:
    #
    # 1 GBP = 105.50 INR

    cleaned["price_inr"] = (
        cleaned["price_gbp"] * GBP_TO_INR
    ).round(2)

    # --------------------------------------------------------
    # SELECT FINAL COLUMNS
    # --------------------------------------------------------

    cleaned = cleaned[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    # Reset index
    cleaned = cleaned.reset_index(drop=True)

    print(
        f"Cleaned dataset contains "
        f"{len(cleaned)} books."
    )

    print(
        f"Number of categories: "
        f"{cleaned['category'].nunique()}"
    )

    return cleaned


# ============================================================
# PHASE 3: SQLITE DATABASE
# ============================================================

def create_database(cleaned_df):

    print("Creating SQLite database...")

    connection = sqlite3.connect("books.db")

    # Enable foreign keys
    connection.execute("PRAGMA foreign_keys = ON")

    cursor = connection.cursor()

    # --------------------------------------------------
    # Create categories table
    # --------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    # --------------------------------------------------
    # Create books table
    # --------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    # --------------------------------------------------
    # Insert ALL categories
    # --------------------------------------------------

    for category in cleaned_df["category"].dropna().unique():

        cursor.execute("""
            INSERT OR IGNORE INTO categories (category_name)
            VALUES (?)
        """, (category,))

    connection.commit()

    # --------------------------------------------------
    # Build category lookup FROM DATABASE
    # --------------------------------------------------

    cursor.execute("""
        SELECT category_id, category_name
        FROM categories
    """)

    category_lookup = {
        category_name: category_id
        for category_id, category_name in cursor.fetchall()
    }

    print("Categories found:")
    print(category_lookup)

    # --------------------------------------------------
    # Insert books
    # --------------------------------------------------

    for _, row in cleaned_df.iterrows():

        category = row["category"]

        # Check category exists before using it
        if category not in category_lookup:

            print(f"Warning: category not found: {category}")
            continue

        category_id = category_lookup[category]

        cursor.execute("""
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            category_id
        ))

    connection.commit()

    print("Books inserted successfully.")

    # --------------------------------------------------
    # Verify
    # --------------------------------------------------

    cursor.execute("SELECT COUNT(*) FROM categories")
    category_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM books")
    book_count = cursor.fetchone()[0]

    print(f"Categories inserted: {category_count}")
    print(f"Books inserted: {book_count}")

    return connection


# ============================================================
# PHASE 4: SQL QUERIES
# ============================================================

def run_sql_queries(connection):
    """
    Execute at least 5 SQL queries demonstrating:
        SELECT
        WHERE
        ORDER BY
        LIMIT
        DISTINCT
        IN/BETWEEN
        JOIN

    Save queries and outputs to query_outputs.txt.
    """

    print("\nRunning SQL queries...")

    queries = {

        # ----------------------------------------------------
        # QUERY 1
        # SELECT + WHERE
        # ----------------------------------------------------

        "Query 1 - Books with rating >= 4": """
            SELECT
                title,
                price_gbp,
                rating,
                in_stock
            FROM books
            WHERE rating >= 4
        """,

        # ----------------------------------------------------
        # QUERY 2
        # ORDER BY + LIMIT
        # ----------------------------------------------------

        "Query 2 - 10 cheapest books": """
            SELECT
                title,
                price_gbp,
                rating
            FROM books
            ORDER BY price_gbp ASC
            LIMIT 10
        """,

        # ----------------------------------------------------
        # QUERY 3
        # DISTINCT
        # ----------------------------------------------------

        "Query 3 - Distinct categories": """
            SELECT DISTINCT
                category_name
            FROM categories
            ORDER BY category_name
        """,

        # ----------------------------------------------------
        # QUERY 4
        # BETWEEN
        # ----------------------------------------------------

        "Query 4 - Books priced between £20 and £40": """
            SELECT
                title,
                price_gbp,
                rating
            FROM books
            WHERE price_gbp BETWEEN 20 AND 40
            ORDER BY price_gbp
        """,

        # ----------------------------------------------------
        # QUERY 5
        # JOIN + ORDER BY + LIMIT
        # ----------------------------------------------------

        "Query 5 - Highest rated books with categories": """
            SELECT
                b.title,
                c.category_name,
                b.rating,
                b.price_gbp,
                b.in_stock
            FROM books b
            JOIN categories c
                ON b.category_id = c.category_id
            ORDER BY b.rating DESC,
                     b.title ASC
            LIMIT 10
        """
    }

    # Open output file
    with open(
        QUERY_OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as output_file:

        for query_name, query in queries.items():

            print("\n" + "=" * 70)
            print(query_name)
            print("=" * 70)

            print(query.strip())

            # Execute SQL
            result = pd.read_sql_query(
                query,
                connection
            )

            print("\nOutput:")
            print(result.to_string(index=False))

            # Save query and output
            output_file.write(
                "\n" + "=" * 70 + "\n"
            )

            output_file.write(
                query_name + "\n"
            )

            output_file.write(
                "=" * 70 + "\n"
            )

            output_file.write(
                query.strip() + "\n\n"
            )

            output_file.write(
                result.to_string(index=False)
            )

            output_file.write("\n")

    print(
        f"\nSQL outputs saved to: "
        f"{QUERY_OUTPUT_PATH}"
    )


# ============================================================
# PHASE 5: PANDAS READ_SQL + MERGE
# ============================================================

def demonstrate_pandas_operations(connection, df):
    """
    Demonstrate:
        1. pd.read_sql()
        2. pd.merge()

    Reproduce the JOIN query using in-memory DataFrames.
    """

    print("\n" + "=" * 70)
    print("PANDAS READ_SQL AND MERGE DEMONSTRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # READ TWO SQL RESULTS INTO DATAFRAMES
    # --------------------------------------------------------

    query_top_rated = """
        SELECT
            title,
            price_gbp,
            rating,
            in_stock
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC
        LIMIT 10
    """

    query_categories = """
        SELECT
            category_id,
            category_name
        FROM categories
        ORDER BY category_name
    """

    df_top_rated = pd.read_sql(
        query_top_rated,
        connection
    )

    df_categories = pd.read_sql(
        query_categories,
        connection
    )

    print("\nDataFrame 1 from pd.read_sql():")
    print(df_top_rated.to_string(index=False))

    print("\nDataFrame 2 from pd.read_sql():")
    print(df_categories.to_string(index=False))

    # --------------------------------------------------------
    # SQL JOIN RESULT USING pd.read_sql()
    # --------------------------------------------------------

    join_query = """
        SELECT
            b.title,
            c.category_name,
            b.rating,
            b.price_gbp,
            b.in_stock
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY b.rating DESC,
                 b.title ASC
        LIMIT 10
    """

    sql_join_df = pd.read_sql(
        join_query,
        connection
    )

    # --------------------------------------------------------
    # REPRODUCE JOIN USING pd.merge()
    # --------------------------------------------------------

    books_for_merge = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        connection
    )

    categories_for_merge = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection
    )

    merged_df = pd.merge(
        books_for_merge,
        categories_for_merge,
        on="category_id",
        how="inner"
    )

    # Select same columns as SQL JOIN
    merged_df = merged_df[
        [
            "title",
            "category_name",
            "rating",
            "price_gbp",
            "in_stock"
        ]
    ]

    # Same sorting and limit as SQL query
    merged_df = (
        merged_df
        .sort_values(
            by=["rating", "title"],
            ascending=[False, True]
        )
        .head(10)
        .reset_index(drop=True)
    )

    sql_join_df = sql_join_df.reset_index(drop=True)

    # --------------------------------------------------------
    # DISPLAY SIDE BY SIDE
    # --------------------------------------------------------

    print("\nSQL JOIN RESULT using pd.read_sql():")
    print(sql_join_df.to_string(index=False))

    print("\npd.merge() RESULT:")
    print(merged_df.to_string(index=False))

    # --------------------------------------------------------
    # CHECK EQUIVALENCE
    # --------------------------------------------------------

    equivalent = sql_join_df.equals(
        merged_df
    )

    print(
        f"\nDo both results match? {equivalent}"
    )

    if not equivalent:

        print(
            "\nThe values may differ because of "
            "data types or floating-point representation."
        )

    else:

        print(
            "\nSUCCESS: pd.read_sql() JOIN and "
            "pd.merge() produce equivalent results."
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 1 - BOOK DATA PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: SCRAPE
    # --------------------------------------------------------

    raw_df = scrape_books()

    print("\nRaw data preview:")
    print(raw_df.head())

    # --------------------------------------------------------
    # STEP 2: CLEAN
    # --------------------------------------------------------

    cleaned_df = clean_data(raw_df)

    print("\nCleaned data preview:")
    print(cleaned_df.head())

    # --------------------------------------------------------
    # VALIDATE REQUIREMENTS
    # --------------------------------------------------------

    print("\nValidation:")
    print(
        f"Book count: {len(cleaned_df)}"
    )

    print(
        f"Category count: "
        f"{cleaned_df['category'].nunique()}"
    )

    print(
        f"Price GBP type: "
        f"{cleaned_df['price_gbp'].dtype}"
    )

    print(
        f"Rating type: "
        f"{cleaned_df['rating'].dtype}"
    )

    print(
        f"In-stock type: "
        f"{cleaned_df['in_stock'].dtype}"
    )

    print(
        f"Price INR type: "
        f"{cleaned_df['price_inr'].dtype}"
    )

    if len(cleaned_df) < 60:
        raise ValueError(
            "Requirement failed: fewer than 60 books."
        )

    if cleaned_df["category"].nunique() < 3:
        raise ValueError(
            "Requirement failed: fewer than 3 categories."
        )

    # --------------------------------------------------------
    # STEP 3: DATABASE
    # --------------------------------------------------------

    connection = create_database(
        cleaned_df
    )

    # --------------------------------------------------------
    # STEP 4: SQL QUERIES
    # --------------------------------------------------------

    run_sql_queries(
        connection
    )

    # --------------------------------------------------------
    # STEP 5: PANDAS OPERATIONS
    # --------------------------------------------------------

    demonstrate_pandas_operations(
        connection,
        cleaned_df
    )

    # --------------------------------------------------------
    # CLOSE DATABASE
    # --------------------------------------------------------

    connection.close()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Database: {DB_PATH}"
    )

    print(
        f"SQL output: {QUERY_OUTPUT_PATH}"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
