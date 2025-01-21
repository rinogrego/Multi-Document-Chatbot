import streamlit as st

def display_knowledge_base():
    st.subheader("PDFs Uploaded", divider = True)
    # display_pdfs_uploaded
    # # print relevant page
    # page_num = 5
    # for doc in documents:
    #     if doc.metadata["page"] == page_num:
    #         page_doc = f"<h4>Page {page_num}</h4>{doc.page_content}"
    #         break
    #     else:
    #         page_doc = "This document don't have that page number"
    # st.write(page_doc, unsafe_allow_html=True)
    # st.markdown("---")
    if len(st.session_state.pdf_titles) == 0:
        no_pdf_html = """
        <div style="display: flex; justify-content: center; align-items: center; color: #ff4b4b; margin-top: 5px;">
            <div style="padding: 5px">
                <p style="font-size: 20px; text-align: center;">No PDF uploaded yet.</p>
                <p style="font-size: 16px; text-align: center;">You can upload your PDF in Add Knowledge tab</p>
            </div>
        </div>
        """
        st.write(no_pdf_html, unsafe_allow_html=True)
    else:
        for title in st.session_state.pdf_titles:
            pdf_html = f"""
            <strong>{title}</strong>
            <p>Pages: {st.session_state.pdf_titles_metadata[title]["num_pages"]}</p>
            <hr>
            """
            st.write(pdf_html, unsafe_allow_html=True)    

    st.markdown("---")
    st.subheader("PubMed Abstracts", divider = True)
    if len(st.session_state.pubmed_papers_keywords) == 0:
        no_pubmed_html = """
        <div style="display: flex; justify-content: center; align-items: center; color: #ff4b4b; margin-top: 5px;">
            <div style="padding: 5px">
                <p style="font-size: 20px; text-align: center;">No PubMed abstracts scrapped yet.</p>
                <p style="font-size: 16px; text-align: center;">You can scrap PubMed abstracts in Add Knowledge tab</p>
            </div>
        </div>
        """
        st.write(no_pubmed_html, unsafe_allow_html=True)
    else:
        # bonus with author info please
        raw_text = ""
        idx = 1
        for search_query in st.session_state.pubmed_papers_keywords: 
            paper_titles = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]["Title"]
            abstracts = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]["Abstract"]
            journals = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]['Journal']
            years = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]['Year']
            months = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]['Month']
            pmids = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]['PMID']
            dois = st.session_state.pubmed_papers_scrap_results[search_query]["df_title_abstracts"]['DOI']
            expander = st.expander(f"**Keywords: {search_query} ({idx}-{idx-1+len(abstracts)})**")
            for title, abs, journal, year, month, pmid, doi in zip(paper_titles, abstracts, journals, years, months, pmids, dois):
                expander.write(
                    f"""<h5>[{idx}] {title}</h5>
                    <p>{abs}</p>
                    <p>
                        Journal : {journal} <br>
                        Date &emsp;&nbsp;: {month}, {year} <br>
                        PMID &emsp;: <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/">{pmid}</a><br>
                        doi&emsp;&emsp;: <a href="https://www.doi.org/{doi}">{doi}</a>
                    </p>
                    <hr>""", 
                    unsafe_allow_html=True
                )
                idx += 1

    st.markdown("---")
    st.subheader("PubMed Articles", divider = True)
    if len(st.session_state.pmcids) == 0:
        no_pubmed_html = """
        <div style="display: flex; justify-content: center; align-items: center; color: #ff4b4b; margin-top: 5px;">
            <div style="padding: 5px">
                <p style="font-size: 20px; text-align: center;">No PubMed articles crawled yet.</p>
                <p style="font-size: 16px; text-align: center;">You can crawl a PubMed article in Add Knowledge tab</p>
            </div>
        </div>
        """
        st.write(no_pubmed_html, unsafe_allow_html=True)
    else:
        expander = st.expander(f"**Total articles: {len(st.session_state.pmcids)}**")
        for pmcid in st.session_state.pmcids:
            pmcid_html = f"""
            <h5>{st.session_state.pmcids_metadata[pmcid]["title"]}</h5>
            <p>PMCID: {pmcid}</p>
            <hr>
            """
            expander.write(pmcid_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Wikipedia Articles", divider = True)
    if len(st.session_state.wikipedia_searches) == 0:
        no_wiki_html = """
        <div style="display: flex; justify-content: center; align-items: center; color: #ff4b4b; margin-top: 5px;">
            <div style="padding: 5px">
                <p style="font-size: 20px; text-align: center;">No Wikipedia articles crawled yet.</p>
                <p style="font-size: 16px; text-align: center;">You can crawl a Wikipedia article in Add Knowledge tab</p>
            </div>
        </div>
        """
        st.write(no_wiki_html, unsafe_allow_html=True)
    else:
        expander = st.expander(f"**Total Keywords: {len(st.session_state.wikipedia_searches)}**")
        for keyword, docs in st.session_state.wikipedia_searches.items():
            wiki_html = f"""<h5>Keywords: {keyword}</h5>"""
            for wiki_document in st.session_state.wikipedia_searches[keyword]:
                title = wiki_document.metadata["title"]
                summary = wiki_document.metadata["summary"]
                source = wiki_document.metadata["source"]
                wiki_document_html = f"""<div><h5>{title}</h5><a href="{source}">Check Source</a></div><br>
                """
                wiki_html += wiki_document_html
            expander.write(wiki_html, unsafe_allow_html=True)