import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


st.set_page_config(
    page_title="Reviews sentiment analysis",
    page_icon="🕸",
    layout="wide",
    initial_sidebar_state="expanded")

def add_bg_color():
    st.markdown(
    f"""
    <style>
    .stApp {{
        primaryColor="green"
	background="#0E1117"
	secondaryBackgroundColor="#262730"
	textColor="#FAFAFA"
	font="sans serif"
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_color()


st.title("Hotel Reviews Sentiment Analysis")
st.markdown("------------------------------------------------------------------------------------")
# asa = st.sidebar.radio('Select company', ('Nokia','Samsung'))

# st.sidebar.markdown("##Upload reviews data :")
#filelocation
DATA_URL= 'Hotel_Reviews_Dataset.csv'
df = pd.read_csv(DATA_URL)
hotel_name = df['Hotel_Name'].unique()
df['year'] = pd.DatetimeIndex(df['Review_Date']).year
year = df['year'].unique()
col1, col2 = st.columns(2)

with col1:
	selected_hotel = st.selectbox('Select Hotel',options = hotel_name,key =1)

with col2:
	selected_year = st.selectbox('Select year',options = year,key =2)
	
data = df[(df['Hotel_Name'] == selected_hotel) & (df['year'] == selected_year)]
data["Reviews"] = data["Reviews"].astype("str")
data["score"] = data["Reviews"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
data["sentiment"] = np.where(data['score'] >= .5, "Positive", "Negative")
data = data[['Reviewer_Nationality','Hotel_Name','Reviews','sentiment','score','Review_Date','year']]
data['Review_Date']=pd.to_datetime(data['Review_Date'])
data['quarter'] = pd.PeriodIndex(data.Review_Date, freq='Q')

st.subheader("Hotel Reviews Sentiment distribution")

col3, col4 = st.columns(2)

with col4:
	sentiment_count = data.groupby(['sentiment'])['sentiment'].count()
	sentiment_count = pd.DataFrame({'Sentiments':sentiment_count.index,'sentiment':sentiment_count.values})
	fig = px.pie(sentiment_count,values='sentiment',names='Sentiments',width=550, 
		height=400).update_layout(title_text='Sentiment distribution', title_x=0.5)
	st.plotly_chart(fig,use_container_width=True)

with col3:
	
	data['Review_Month'] = data['Review_Date'].dt.strftime('%m-%Y')
	trend_dt1 = data.groupby(['Review_Month','sentiment']).size().reset_index()
	trend_dt1 = trend_dt1.sort_values(['sentiment'],ascending=False)
	trend_dt1.rename(columns = {0:'Sentiment_Count'}, inplace = True)

	fig2 = px.line(trend_dt1, x="Review_Month", y="Sentiment_Count", color='sentiment',width=600, 
		height=400).update_layout(title_text='Trend analysis of sentiments', title_x=0.5)
	st.plotly_chart(fig2,use_container_width=True)

st.markdown("------------------------------------------------------------------------------------")

st.subheader("Word Cloud for Reviews Sentiment")

word_ls = ['Room Size','room,','cleanliness','staff','food','AC','expensive','location','service','bathroom','noise','noisy','free','lounge','comfort','comfortable','swimming pool','dirty','overpriced','ambiance','check-in','check in','check out','check-out','toilet','affordable','lift','smell','complimentary','professional','friendly','amazing','good','excellent','bad','pathetic','disappoint','rude','negative','cramped','broken','uncomfortable','positive']

data['Reviews1'] = data['Reviews'].apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in (word_ls)]))

col5, col6 = st.columns(2)

with col5:
	# st.text("Positive reviews word cloud")
	st.set_option('deprecation.showPyplotGlobalUse', False)
	df = data[(data["sentiment"]=="Positive") & (df['Hotel_Name'] == selected_hotel) & (df['year'] == selected_year) & (data['score'] > .8)]
	words = " ".join(df["Reviews1"])
	wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(words)
	plt.imshow(wordcloud)
	plt.xticks([])
	plt.yticks([])
	plt.title("Positive Reviews Word Cloud")
	st.pyplot()

with col6:        
	# st.text("Negative reviews word cloud")
	st.set_option('deprecation.showPyplotGlobalUse', False)
	df = data[(data["sentiment"]=="Negative") & (df['Hotel_Name'] == selected_hotel) & (df['year'] == selected_year) & (data['score'] <=.4)]
	words = " ".join(df["Reviews1"])
	wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640,colormap="RdYlGn").generate(words)
	plt.imshow(wordcloud)
	plt.xticks([])
	plt.yticks([])
	plt.title("Negative Reviews Word Cloud")
	st.pyplot()
	
st.markdown("------------------------------------------------------------------------------------")
	
   
