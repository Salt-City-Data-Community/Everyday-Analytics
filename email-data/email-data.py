from email.header import decode_header, make_header
import mailbox
import nltk
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import pandas as pd

subjects = []

def get_subject_data(mbox, output):
  # loop to make dictionary based on id, subject, date, sender
    for message in mbox:
        id = message['message-id']
        subject = str(make_header(decode_header(message['subject'])))
        date = message['date']
        sender = str(make_header(decode_header(message['from'])))
        output.append({'id': id, 'subject': subject, 'date': date, 'sender': sender})
    return output

mbox = mailbox.mbox('oct-sample.mbox')

output = get_subject_data(mbox, subjects)

# check keys with below 2 lines
#message=mbox[0]
#print(message.keys()) # ['Date'] and ['From'] will be used

df = pd.DataFrame(output)

# frequency distribution of tokens from subject column of df
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# tokenize subject column
df['subject_tokens'] = df['subject'].apply(word_tokenize)

# remove stop words
df['subject_tokens'] = df['subject_tokens'].apply(lambda x: [item for item in x if item not in stop_words])

word_list = ' '.join(df['subject_tokens'].apply(lambda x: ' '.join(x)))
#lowercase all words in list
word_list_lower = word_list.lower()
word_tokens = nltk.word_tokenize(word_list_lower)

# frequency distribution of tokens
fdist = FreqDist(word_tokens)
print(fdist.most_common(15))