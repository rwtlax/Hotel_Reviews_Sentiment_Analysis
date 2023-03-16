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

per_dt = data.groupby(['Reviewer_Nationality','sentiment']).size().reset_index()
per_dt = per_dt.sort_values(['sentiment'],ascending=True)
per_dt1 = data.groupby(['Reviewer_Nationality']).size().reset_index()
per_dt2 = pd.merge(per_dt,per_dt1,how = 'left', on = 'Reviewer_Nationality')
per_dt2['Sentiment_Percentage'] = per_dt2['0_x']/per_dt2['0_y']
per_dt2 = per_dt2[['Reviewer_Nationality','sentiment','Sentiment_Percentage']]


st.subheader("Hotel Reviews Sentiment distribution")

col3, col4 = st.columns(2)

with col4:
	sentiment_count = data.groupby(['sentiment'])['sentiment'].count()
	sentiment_count = pd.DataFrame({'Sentiments':sentiment_count.index,'sentiment':sentiment_count.values})
	fig = px.pie(sentiment_count,values='sentiment',names='Sentiments',color_discrete_sequence=["blue", "green"],width=550, 
		height=400).update_layout(title_text='Sentiment distribution', title_x=0.5)
	st.plotly_chart(fig,use_container_width=True)

with col3:
	
	data['Review_Month'] = data['Review_Date'].dt.strftime('%m-%Y')
	trend_dt1 = data.groupby(['Review_Month','sentiment']).size().reset_index()
	trend_dt1 = trend_dt1.sort_values(['Review_Month'],ascending=True)
	trend_dt1.rename(columns = {0:'Sentiment_Count'}, inplace = True)

	fig2 = px.line(trend_dt1, x="Review_Month", y="Sentiment_Count",color='sentiment',width=600, 
		height=400).update_layout(title_text='Trend analysis of sentiments', title_x=0.5)
	st.plotly_chart(fig2,use_container_width=True)

st.markdown("------------------------------------------------------------------------------------")
fig = px.histogram(data, x="Reviewer_Nationality", y="sentiment",
	histfunc="count", color="sentiment",facet_col="sentiment", 
	labels={"sentiment": "sentiment"},width=550, height=400).update_layout(title_text='Distribution by count of sentiment', title_x=0.5)
st.plotly_chart(fig,use_container_width=True)
	
st.markdown("------------------------------------------------------------------------------------")

fig1 = px.histogram(per_dt2, x= "Reviewer_Nationality", y="Sentiment_Percentage",color="sentiment" ,facet_col="sentiment", labels={"sentiment": "sentiment"},
width=550, height=400).update_layout(yaxis_title="Percentage",title_text='Distribution by percentage of sentiment', title_x=0.5)
st.plotly_chart(fig1,use_container_width=True)
	
st.markdown("------------------------------------------------------------------------------------")

st.subheader("Word Cloud for Reviews Sentiment")

word_ls = ['Room Size','room,','cleanliness','staff','food','AC','expensive','location','service','bathroom','noise',
	   'noisy','free','lounge','will','never','really','comfort','comfortable','swimming pool','dirty','overpriced','ambiance','check-in',
	   'check in','check out','check-out','toilet','affordable','lift','smell','complimentary','professional','friendly','amazing','good','excellent',
	   'bad','pathetic','disappoint','rude','negative','cramped','broken','uncomfortable']

data['Reviews1'] = data['Reviews'].apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in (word_ls)]))
data['Reviews1'] = data['Reviews1'].str.replace('hotel',' ')

col5, col6 = st.columns(2)

with col5:
	# st.text("Positive reviews word cloud")
	st.set_option('deprecation.showPyplotGlobalUse', False)
	df1 = data[(data["sentiment"]=="Positive") & (data['score'] > .8)]
	words = " ".join(df1["Reviews1"])
	wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(words)
	plt.imshow(wordcloud)
	plt.xticks([])
	plt.yticks([])
	plt.title("Positive Reviews Word Cloud")
	st.pyplot()

with col6:        
	# st.text("Negative reviews word cloud")
	st.set_option('deprecation.showPyplotGlobalUse', False)
	df2 = data[(data["sentiment"]=="Negative")  & (data['score'] <=.2)]
	words = " ".join(df2["Reviews1"])
	wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640,colormap="RdYlGn").generate(words)
	plt.imshow(wordcloud)
	plt.xticks([])
	plt.yticks([])
	plt.title("Negative Reviews Word Cloud")
	st.pyplot()
	
st.markdown("------------------------------------------------------------------------------------")

st.subheader("Top 5 Positive Reviews :")

pos = data[(data['score'] > .8)].reset_index()
pos = pos.sort_values(['score'],ascending=False)
st.write("1. Sentiment Score: " +str(pos['score'][6]) + " - " + str(pos['Reviews'][6]))
st.write("2. Sentiment Score: " +str(pos['score'][7]) + " - " + str(pos['Reviews'][7]))
st.write("3. Sentiment Score: " +str(pos['score'][8]) + " - " + str(pos['Reviews'][8]))
st.write("4. Sentiment Score: " +str(pos['score'][9]) + " - " + str(pos['Reviews'][9]))
st.write("5. Sentiment Score: " +str(pos['score'][10]) + " - " + str(pos['Reviews'][10]))


st.markdown("------------------------------------------------------------------------------------")
st.subheader("Top 5 Negative Reviews:")

neg = data[(data['score'] < .1)].reset_index()


st.markdown("1. Sentiment Score: " +str(neg['score'][1]) + " - " + str(neg['Reviews'][1]))
st.markdown("2. Sentiment Score: " +str(neg['score'][2]) + " - " + str(neg['Reviews'][2]))
st.markdown("3. Sentiment Score: " +str(neg['score'][3]) + " - " + str(neg['Reviews'][3]))
st.markdown("4. Sentiment Score: " +str(neg['score'][4]) + " - " + str(neg['Reviews'][4]))
st.markdown("5. Sentiment Score: " +str(neg['score'][5]) + " - " + str(neg['Reviews'][5]))

st.markdown("------------------------------------------------------------------------------------")
st.subheader("Enter your reviews for the hotel:")

text = st.text_area('Type here', ''' ''')
btn_pressed = st.button('Sentiment Score')

def getSentimentScore(txt):
	if txt is not None:
		sent_score = analyzer.polarity_scores(txt)
		if sent_score["compound"] >= .7:
			st.text('Review is Positive. Sentiment score is: '+str(sent_score["compound"]))
		elif (sent_score["compound"] < .7) and (sent_score["compound"] > .1):
			st.text('Review is Neutral. Sentiment score is: '+str(sent_score["compound"]))
		else:
			st.text('Review is Negative. Sentiment score is: '+str(sent_score["compound"]))
	else:
		st.text('Please enter some review about a hotel.')
if btn_pressed:
	getSentimentScore(text)
	
		

