# streamlit run app.py
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image

session = st.sidebar.selectbox("Section", ["Introduction", "Data Analysis", "Prediction"])
st.title('Berry Yield Prediction')
df = pd.read_excel('sample.xlsx', header=None, names=['y'])
df['Week'] = list(range(1, df.shape[0] + 1))
yhats = pd.read_csv('yhat.csv')

if session == "Introduction":
    st.sidebar.subheader("Dashboard Introduction")

    # image
    image = Image.open('berry_img.jpg')
    st.image(image, width=700)
    st.subheader("Introduction")
    st.write("""
    This is a dashboard aiming to analyze the berry yield dataset recorded on a weekly basis.
    The major goal of this project is to predict the berry yield in the next 10 weeks.
    The original dataset only contains the production information, and no other information like the dates of records is provided.
    Being a typical time series problem with apparent seasonal trend, Seasonal Autoregressive Integrated Moving Average (SARIMA) and Facebook Prophet are applied to get the prediction results.
    After the model selection process based on the MSE in the validation set, SARIMA is decided as the final model.
    The prediction results of both models will be exhibited in this dashboard.

    This dashboard includes the following three parts:
    - Introduction
    - Data Analysis
    - Prediction

    The **Data Analysis** section uses the original data to create visualizations and conduct analysis, while the predicted values are incorporated for analysis in the **Prediction** part.""")

if session == "Data Analysis":
    # sidebar
    st.sidebar.subheader("Data Analysis")
    parts = st.sidebar.radio("Two Parts:", ["Overview", "Periodic Data"])

    if parts == "Overview":
        # sidebar
        st.sidebar.subheader("Overview")
        week_range = st.sidebar.slider('Range of week:', 1, max(df['Week']), (1, max(df['Week'])))

        # Reference: https://altair-viz.github.io/gallery/multiline_tooltip.html
        # Create a selection that chooses the nearest point & selects based on x-value
        source = df[df['Week'].isin(list(range(week_range[0], week_range[1] + 1)))]
        source['Category'] = 'True Production'
        source['Production'] = np.round(source['y'], 3)
        # Create a selection that chooses the nearest point & selects based on x-value
        nearest = alt.selection(type='single', nearest=True, on='mouseover',
                                fields=['Week'], empty='none')

        # The basic line
        line = alt.Chart(source).mark_line(interpolate='basis').encode(
            x='Week:Q',
            y='Production:Q',
            color='Category:N'
        )

        # Transparent selectors across the chart. This is what tells us
        # the x-value of the cursor
        selectors = alt.Chart(source).mark_point().encode(
            x='Week:Q',
            opacity=alt.value(0),
        ).add_selection(
            nearest
        )

        # Draw points on the line, and highlight based on selection
        points = line.mark_point().encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )

        # Draw text labels near the points, and highlight based on selection
        text = line.mark_text(align='left', dx=5, dy=-5).encode(
            text=alt.condition(nearest, 'Production:Q', alt.value(' '))
        )

        # Draw a rule at the location of the selection
        rules = alt.Chart(source).mark_rule(color='gray').encode(
            x='Week:Q',
        ).transform_filter(
            nearest
        )

        # Put the five layers into a chart and bind the data
        fig1 = alt.layer(
            line, selectors, points, rules, text
        ).properties(
            width=600, height=300
        )

        # show figure
        ex = st.checkbox('See Explanation')
        if ex:
            st.write("""This is the visualization of the original weekly dataset, which describes the general trend of berry yield.
            By adjusting the slider, users can select a specific time range to study the berry yield fluctuation within that period.
            Based on the figure, there is a consistent seasonal pattern on berry yield.""")
        st.subheader('Weekly Berry Yield')
        st.altair_chart(fig1, use_container_width=True)

    if parts == "Periodic Data":
        st.sidebar.subheader("Periodic Data")
        week_start = st.sidebar.number_input("Starting Week (up to 52):", 1, 52, 1)

        # data
        source = df[df['Week'] >= week_start]
        source['Week'] = source['Week'] - week_start + 1
        source_s = pd.Series(np.concatenate((np.repeat('1st Year', 52),
                                             np.repeat('2nd Year', 52), np.repeat('3rd Year', 52),
                                             np.repeat('4th Year', 52),
                                             np.repeat('5th Year', df.shape[0] - 52 * 4))), name='Time')
        source.reset_index(drop=True, inplace=True)
        source = pd.concat([source, source_s[:source.shape[0]]], axis=1)
        source['Position Within the Cycle'] = (source['Week'] - 1) % 52 + 1

        years = st.sidebar.multiselect("Included Years:", list(source['Time'].unique()),
                                       default=list(source['Time'].unique()))
        # figure
        source = source[source['Time'].isin(years)]
        fig4 = alt.Chart().mark_line(point=True).encode(
            alt.X('Position Within the Cycle'),
            alt.Y('y', title='Production'),
            # alt.Facet('Time', columns=1),
        )
        fig4 = alt.layer(fig4, data=source).facet(
            row='Time'
        )

        # show figure
        ex = st.checkbox('See Explanation')
        if ex:
            st.write("""This figure breaks the original dataset into separate year by approximating one year as 52 weeks.
            The sidebars allow the users to determine the starting week and included years.
            The berry yield data before the starting week and outside of the included years will not show in the figure.
            By reading this chart, the weekly data at the same position in each cycle can be easily compared.
            For example, we can see that the cycle peak is moving across time.""")
        st.subheader('Weekly Berry Yield in Each Year')
        st.altair_chart(fig4, use_container_width=True)

if session == "Prediction":
    model = st.sidebar.selectbox("Predictive Model", ["SARIMA", "Prophet"])
    if model == "SARIMA":
        yhat = yhats['yhat']
    else:
        yhat = yhats['yhat_p']
    yhat = pd.Series(np.array(yhat), name='yhat')
    # data
    st.sidebar.subheader("Prediction")
    df['Predicted Production'] = yhat[:-10]
    source = df.rename({'y': 'True Production'}, axis=1).melt('Week', var_name='Category', value_name='Production')
    df_pred = pd.DataFrame(yhat[-10:]).reset_index().rename({'index': 'Week', 'yhat': 'Production'}, axis=1)
    df_pred['Week'] = df_pred['Week'] + 1
    source = pd.concat([source, df_pred.assign(Category='Predicted Production')], axis=0)
    parts = st.sidebar.radio("Three Parts:", ["Prediction Result", "Prediction Comparison", "Periodic Data"])
    if parts == 'Prediction Result':
        # sidebar
        st.sidebar.subheader("Prediction Result")
        week_range = st.sidebar.slider('Range of week:', 1, max(source['Week']), (1, max(source['Week'])))
        plots = st.sidebar.multiselect("Show Chart:", ['True Production', 'Predicted Production'],
                                       default=['True Production', 'Predicted Production'])

        # plot
        source = source[source['Week'].isin(list(range(week_range[0], week_range[1] + 1)))]
        source = source[source['Category'].isin(plots)]
        source = np.round(source, 3)
        # Create a selection that chooses the nearest point & selects based on x-value
        nearest = alt.selection(type='single', nearest=True, on='mouseover',
                                fields=['Week'], empty='none')

        # The basic line
        line = alt.Chart(source).mark_line(interpolate='basis').encode(
            x='Week:Q',
            y='Production:Q',
            color='Category:N'
        )

        # Transparent selectors across the chart. This is what tells us
        # the x-value of the cursor
        selectors = alt.Chart(source).mark_point().encode(
            x='Week:Q',
            opacity=alt.value(0),
        ).add_selection(
            nearest
        )

        # Draw points on the line, and highlight based on selection
        points = line.mark_point().encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )

        # Draw text labels near the points, and highlight based on selection
        text = line.mark_text(align='left', dx=5, dy=-5).encode(
            text=alt.condition(nearest, 'Production:Q', alt.value(' '))
        )

        # Draw a rule at the location of the selection
        rules = alt.Chart(source).mark_rule(color='gray').encode(
            x='Week:Q',
        ).transform_filter(
            nearest
        )

        # Add vertical line
        if week_range[1] >= df.shape[0]:
            overlay = pd.DataFrame({'x': [df.shape[0]]})
            vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=2, strokeDash=[3, 5]).encode(x='x:Q')
            # Put the five layers into a chart and bind the data
            fig2 = alt.layer(
                line, selectors, points, rules, text, vline
            ).properties(
                width=600, height=300
            )
        else:
            # Put the five layers into a chart and bind the data
            fig2 = alt.layer(
                line, selectors, points, rules, text
            ).properties(
                width=600, height=300
            )

        # show figure
        ex = st.checkbox('See Explanation')
        if ex:
            st.write("""The model selection procedure can be accessed at the Jupyter notebook.
            The model applied currently in this section is {}.
            Parameter tuning has been conducted for the final model. The goal is to predict the berry yield in the next 10 weeks.
            The exact predicted data is shown in the table at bottom, where the week IDs are represented by their index in the original dataset starting from 1.""".format(
                model))
        st.subheader('Predicted and True Berry Yield on a Weekly Basis')
        st.altair_chart(fig2, use_container_width=True)
        check = st.checkbox('Show the Next 10 Weeks of Productions Using {}'.format(model), value=False)
        if check:
            st.write(df_pred.rename({'Production': 'Predicted Production'}, axis=1))

    if parts == 'Prediction Comparison':
        st.sidebar.subheader("Prediction Comparison")
        weeks = st.sidebar.multiselect('Predicted Weeks:', list(df_pred['Week']), default=[236, 237])

        # data
        ddf = np.zeros((df_pred.shape[0], 4))
        for i in range(df_pred.shape[0]):
            for j in range(4):
                ddf[i, j] = df[df['Week'] == 236 + i - 52 * (4 - j)]['y']
        data = pd.concat([df_pred, pd.DataFrame(ddf, columns=['1st Year', '2nd Year', '3rd Year', '4th Year'])],
                         axis=1).rename({'Production': '5th Year'}, axis=1)
        data = data.melt('Week', var_name='Time', value_name='Production')
        data = data[data['Week'].isin(weeks)]

        fig3 = alt.Chart(data).mark_bar(size=20).encode(
            x='Week:N',
            y='Production:Q',
            color='Week:N',
            column='Time:N'
        )

        # figure
        ex = st.checkbox('See Explanation')
        if ex:
            st.write("""In this section, the goal is to compare the predicted berry yield values in the next 10 weeks with their counterparts in the previous years.
            The week IDs starts from 236 and ends with 245. The approximate trend of yield can be obtained through this figure.""")
        st.subheader('Berry Yield Comparison for Each Year')
        st.write(fig3)
        check = st.checkbox('Show the Next 10 Weeks of Productions Using {}'.format(model), value=False)
        if check:
            st.write(df_pred.rename({'Production': 'Predicted Production'}, axis=1))

    if parts == "Periodic Data":
        st.sidebar.subheader("Periodic Data")
        week_start = st.sidebar.number_input("Starting Week (up to 52):", 1, 52, 1)

        # data
        source = df[df['Week'] >= week_start]
        source = pd.concat([source, df_pred.rename({'Production': 'y'}, axis=1)], axis=0)
        source['Week'] = source['Week'] - week_start + 1
        source_s = pd.Series(np.concatenate((np.repeat('1st Year', 52),
                                             np.repeat('2nd Year', 52), np.repeat('3rd Year', 52),
                                             np.repeat('4th Year', 52),
                                             np.repeat('5th Year', df.shape[0] - 52 * 4 + 10))), name='Time')
        source.reset_index(drop=True, inplace=True)
        source = pd.concat([source, source_s[:source.shape[0]]], axis=1)
        source['Position Within the Cycle'] = (source['Week'] - 1) % 52 + 1

        years = st.sidebar.multiselect("Included Years:", list(source['Time'].unique()),
                                       default=list(source['Time'].unique()))
        # figure
        source = source[source['Time'].isin(years)]
        fig5 = alt.Chart().mark_line(point=True).encode(
            alt.X('Position Within the Cycle'),
            alt.Y('y', title='Production'),
        )
        overlay = pd.DataFrame({'x': [source.iloc[-11, :]['Position Within the Cycle']]})
        vline = alt.Chart(overlay).mark_rule(color='black', strokeWidth=2, strokeDash=[3, 5]).encode(x='x:Q')

        fig5 = alt.layer(fig5, vline, data=source).facet(
            row='Time'
        )

        # show figure
        ex = st.checkbox('See Explanation')
        if ex:
            st.write("""The objective for this figure is to detect whether the yield trend in the last year accords with the corresponding trends in the previous years.
            The black dotted vertical lines in each figure denote the counterpart of the starting week of prediction.
            Similar to the Data Analysis section, starting week and included years can be adjusted using the sidebars.""")
        st.subheader('Weekly Berry Yield in Each Year Including the Predicted Values')
        st.altair_chart(fig5, use_container_width=True)
        check = st.checkbox('Show the Next 10 Weeks of Productions Using {}'.format(model), value=False)
        if check:
            st.write(df_pred.rename({'Production': 'Predicted Production'}, axis=1))