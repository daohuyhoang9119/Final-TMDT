import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from funcs import *
import streamlit.components.v1 as components

def main():

    st.set_page_config(layout="wide", initial_sidebar_state='expanded')
    github = """
    <a href="https://github.com/mnobeidat13", target="_blank">
      <img src="https://pbs.twimg.com/profile_images/1414990564408262661/r6YemvF9_400x400.jpg" alt="HTML tutorial" style="width:50px;height:50px;">

    </a>
    """

    kaggle = """
    <a href="https://www.kaggle.com/mohammedobeidat", target="_blank">
      <img src="https://miro.medium.com/max/3200/1*K5NPQiLmq30qmkySiVb5JQ.jpeg" alt="HTML tutorial" style="width:100px;height:50px;">

    </a>
    """

    linkedin = """
    <a href="https://www.linkedin.com/in/mnobeidat/", target="_blank">
      <img src="https://play-lh.googleusercontent.com/kMofEFLjobZy_bCuaiDogzBcUT-dz3BBbOrIEjJ-hqOabjK8ieuevGe6wlTD15QzOqw" alt="HTML tutorial" style="width:50px;height:50px;">

    </a>
    """
    
        
    sidebar_header = '''This is a demo to illustrate a recommender system that finds similar items to a given clothing article or recommend items for a customer using 4 different approaches:'''
    
    page_options = ["Find similar items",
                    "Customer Recommendations",]
    
    st.sidebar.info(sidebar_header)


    
    page_selection = st.sidebar.radio("Try", page_options)
    articles_df = pd.read_csv('articles.csv')
    
    models = ['Similar items based on image embeddings', 
              'Similar items based on embeddings from TensorFlow Recommendrs model',
              ]
    
    model_descs = ['Image embeddings are calculated using VGG16 CNN from Keras', 
                  'TFRS model performes a collaborative filtering based ranking using a neural network', 
                  ]

#########################################################################################
#########################################################################################

    if page_selection == "Find similar items":

        articles_rcmnds = pd.read_csv('results/articles_rcmnds.csv')

        articles = articles_rcmnds.article_id.unique()
        get_item = st.sidebar.button('Get Random Item')
        
        if get_item:
            
            rand_article = np.random.choice(articles)

            article_data = articles_rcmnds[articles_rcmnds.article_id == rand_article]
            rand_article_desc = articles_df[articles_df.article_id == rand_article].detail_desc.iloc[0]
            image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds = get_rcmnds(article_data)
            
            rcmnds = (image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            scores = get_rcmnds_scores(article_data)
            features = get_rcmnds_features(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            images = get_rcmnds_images(image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            detail_descs  = get_rcmnds_desc(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            st.sidebar.image(get_item_image(str(rand_article), width=200, height=300))
            st.sidebar.write('Article description')
            st.sidebar.caption(rand_article_desc)

            with st.container():     
                for i, model, image_set, score_set, model_desc, detail_desc_set, features_set, rcmnds_set in zip(range(5), models, images, scores, model_descs, detail_descs, features, rcmnds):
                    container = st.expander(model, expanded = model == 'Similar items based on image embeddings' or model == 'Similar items based on text embeddings')
                    with container:
                        cols = st.columns(7)
                        cols[0].write('###### Similarity Score')
                        cols[0].caption(model_desc)
                        for img, col, score, detail_desc, rcmnd in zip(image_set[1:], cols[1:], score_set[1:], detail_desc_set[1:],  rcmnds_set[1:]):
                            with col:
                                st.caption('{}'.format(score))
                                st.image(img, use_column_width=True)
                                if model == 'Similar items based on text embeddings':
                                    st.caption(detail_desc)
                                    
#########################################################################################
#########################################################################################

    
#########################################################################################
#########################################################################################
    if page_selection == "Customer Recommendations":
        
        customers_rcmnds = pd.read_csv('results/customers_rcmnds.csv')
        customers = customers_rcmnds.customer.unique()        
        
        get_item = st.sidebar.button('Get Random Customer')
        if get_item:
            st.sidebar.write('#### Customer history')

            rand_customer = np.random.choice(customers)
            customer_data = customers_rcmnds[customers_rcmnds.customer == rand_customer]
            customer_history = np.array(eval(customer_data.history.iloc[0]))

            image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds = get_rcmnds(customer_data)
            
            scores = get_rcmnds_scores(customer_data)
            features = get_rcmnds_features(articles_df, combined_rcmnds, tfrs_rcmnds, image_rcmnds, text_rcmnds, feature_rcmnds)
            images = get_rcmnds_images(combined_rcmnds, tfrs_rcmnds, image_rcmnds, text_rcmnds, feature_rcmnds)
            detail_descs  = get_rcmnds_desc(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            rcmnds = (image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)

            splits = [customer_history[i:i+3] for i in range(0, len(customer_history), 3)]
                            
            for split in splits:
                with st.sidebar.container():
                    cols = st.columns(3)
                    for item, col in zip(split, cols):
                        col.image(get_item_image(str(item), 100))
                    

            with st.container():            
                for i, model, image_set, score_set, model_desc, detail_desc_set, features_set, rcmnds_set in zip(range(5), models, images, scores, model_descs, detail_descs, features, rcmnds):
                    container = st.expander(model, expanded=True)
                    with container:
                        cols = st.columns(7)
                        cols[0].write('###### Similarity Score')
                        cols[0].caption(model_desc)
                        for img, col, score, detail_desc, rcmnd in zip(image_set[1:], cols[1:], score_set[1:], detail_desc_set[1:],  rcmnds_set[1:]):
                            with col:
                                st.caption('{}'.format(score))
                                st.image(img, use_column_width=True)
                                

#########################################################################################  
#########################################################################################

    
if __name__ == '__main__':
    main()
