import numpy as np
import pandas as pd

# Enter local paths to CSV files below
books_df = pd.read_csv("data/Books.csv", encoding = "ISO-8859-1")
events_df = pd.read_csv("data/UserEvents.csv", encoding = "ISO-8859-1")
users_df = pd.read_csv("data/Users.csv", encoding = "ISO-8859-1")

def clean_users():
    # Replace missing user ids with values from 'Unnamed: 0' since they are integer versions of user ids
    users_df.user = users_df.loc[:]['Unnamed: 0']

    # Get rid of inaccurate ages
    users_df.loc[(users_df.age > 120), 'age'] = np.nan
    users_df.age = users_df.age.fillna(users_df.age.mean())
    users_df.age = users_df.age.astype(np.int32)

    return users_df

def clean_books():
    # We remove the unnecessary columns from the books dataset
    books_df.drop(['Unnamed: 0', 'urlId'], axis=1, inplace=True)

    # Correct the necessary rows
    books_df.loc[books_df.bookISBN == '078946697X', 'yearOfPublication'] = 2000
    books_df.loc[books_df.bookISBN == '078946697X', 'author'] = "Michael Teitelbaum"
    books_df.loc[books_df.bookISBN == '078946697X', 'publisher'] = "DK Publishing Inc"
    books_df.loc[books_df.bookISBN == '078946697X', 'bookName'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"

    books_df.loc[books_df.bookISBN == '0789466953', 'yearOfPublication'] = 2000
    books_df.loc[books_df.bookISBN == '0789466953', 'author'] = "James Buckley"
    books_df.loc[books_df.bookISBN == '0789466953', 'publisher'] = "2000"
    books_df.loc[books_df.bookISBN == '0789466953', 'bookName'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

    # Convert the years from strings to integers
    books_df.yearOfPublication = pd.to_numeric(books_df.yearOfPublication, errors='coerce')

    # We will assume that 0 and years after 2018 are invalid
    books_df.loc[(books_df.yearOfPublication > 2018) | (books_df.yearOfPublication == 0), 'yearOfPublication'] = np.NAN
    books_df.yearOfPublication.fillna(round(books_df.yearOfPublication.mean()), inplace=True)
    books_df.yearOfPublication = books_df.yearOfPublication.astype(np.int32)

    # Replacing missing publisher with 'other'
    books_df.loc[(books_df.bookISBN == '1931696993'), 'publisher'] = 'other'

def create_csv():
    impressions = events_df.as_matrix(['user', 'bookId', 'impression'])
    books = clean_books()
    users = clean_users()

    # Events dateframe should have userId and bookId which exist in the respective dataframes
    events_new = events_df[events_df.bookId.isin(books.bookISBN)]
    events_new = events_new[events_df.user.isin(users.user)]

    # Fill cells currently with no impression
    events_new.impression.astype(object).fillna("impression")

    # Convert impression strings to integers, we start at 1 so we can label null values 0
    mapping = {'dislike': 1, 'like': 2, 'add to cart': 3, 'view': 4, 'interact': 5, 'checkout': 6}
    impressions = events_new.replace({'impression': mapping})

    # Create a user x bookId matrix
    impressions_matrix = impressions.pivot(index='user', columns='bookId', values='impression')
    user = impressions_matrix.index
    bookId = impressions_matrix.columns

    # Fill the NaN values with 0
    impressions_matrix = impressions_matrix.astype(object).fillna(0)
    impressions_matrix = impressions_matrix.astype(np.int32)

    # Save matrix to CSV file
    impressions_matrix.to_csv('impressions_matrix.csv')

if __name__ == '__main__':
    create_csv()

