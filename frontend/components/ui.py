
import streamlit as st


def product_card(title, brand="", price="", image=None, url=""):
    with st.container(border=True):
        if image:
            st.image(image, use_container_width=True)

        st.markdown(f"**{title}**")

        if brand:
            st.caption(brand)

        if price:
            st.write(price)

        if url and str(url).startswith("http"):
            st.link_button("View Details", url, use_container_width=True)
