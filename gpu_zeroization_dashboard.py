from pathlib import Path
import json
import subprocess
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
st.set_page_config(page_title='GPU Zeroization Lab', page_icon='🛡️', layout='wide')
st.markdown('''<style> .block-container {max-width:1100px;padding-top:2rem;} h1,h2,h3{letter-spacing:-.025em;} [data-testid="stMetric"]{background:rgba(125,125,125,.06);padding:14px;border-radius:10px;} </style>''', unsafe_allow_html=True)

@st.cache_data
def csv(name):
    return pd.read_csv(RESULTS / name, encoding='utf-8')
@st.cache_data
def js(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))
@st.cache_data
def jsonl(name):
    with (RESULTS / name).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def verdict_counts(df):
    return df['checker_verdict'].value_counts().rename_axis('Verdict').reset_index(name='Count')

st.title('GPU Zeroization Lab')
st.caption('Static GPU instruction analysis · Research demonstration · Local evidence viewer')
if not (RESULTS / 'classifications.csv').exists():
    st.error('Place this dashboard file in the repository root, beside the results/ folder.')
    st.stop()

cls = csv('classifications.csv')
features = {r['artifact_id']: r for r in jsonl('features.jsonl')}
artifacts = {r['artifact_id']: r for r in jsonl('artifacts.jsonl')}
tabs = st.tabs(['Overview', 'Classification explorer', 'Evidence & validation', 'Live tests'])

with tabs[0]:
    st.subheader('Research question')
    st.write('Do GPU memory-zeroization instruction signatures survive compilation across CUDA constructs, contexts, architectures, and optimization settings?')
    a,b,c = st.columns(3)
    a.metric('Classified records', f'{len(cls):,}')
    b.metric('GPU architectures', cls['target_arch'].nunique())
    c.metric('CUDA versions', cls['cuda_version'].nunique())
    counts=verdict_counts(cls)
    fig=px.bar(counts,x='Verdict',y='Count',text='Count',color='Verdict',color_discrete_map={'FULL_SIGNATURE':'#238b68','PARTIAL_SIGNATURE':'#d39a38','STORE_ONLY':'#4c82ba','ABSENT':'#bd5c62','UNDECIDABLE':'#777777'})
    fig.update_layout(showlegend=False,margin=dict(l=10,r=10,t=10,b=10),height=370)
    st.plotly_chart(fig,use_container_width=True)
    st.info('FULL_SIGNATURE means the expected static instruction ordering was detected. It does not establish secure erasure on every dynamic execution path.')
    with st.expander('What do the verdicts mean?'):
        st.markdown('''**FULL_SIGNATURE** — expected zero-store, CTA memory barrier, synchronization barrier, and exit ordering detected.\n\n**PARTIAL_SIGNATURE** — some required instruction evidence exists, but the full ordering or components are missing.\n\n**STORE_ONLY** — zero-valued shared store detected without the required barrier combination.\n\n**ABSENT** — no qualifying zero-valued shared store detected.\n\n**UNDECIDABLE** — available static evidence is insufficient for a confident classification.''')

with tabs[1]:
    st.subheader('Explore recorded configurations')
    col1,col2,col3,col4 = st.columns(4)
    filters=[('zeroization_construct',col1,'Construct'),('kernel_context',col2,'Context'),('target_arch',col3,'Architecture'),('ptxas_optimization',col4,'Optimization')]
    filtered=cls.copy()
    for field,column,label in filters:
        values=sorted(cls[field].dropna().unique().tolist())
        selected=column.selectbox(label,['All']+values,format_func=str,key='filter_'+field)
        if selected!='All': filtered=filtered[filtered[field]==selected]
    verdict=st.selectbox('Verdict',['All']+sorted(cls['checker_verdict'].dropna().unique().tolist()))
    if verdict!='All': filtered=filtered[filtered['checker_verdict']==verdict]
    st.caption(f'{len(filtered):,} matching records')
    columns=['artifact_id','zeroization_construct','kernel_context','target_arch','cuda_version','ptxas_optimization','checker_verdict']
    st.dataframe(filtered[columns],hide_index=True,use_container_width=True,height=230)
    if filtered.empty:
        st.warning('No records match the selected filters.')
    else:
        aid=st.selectbox('Inspect artifact',filtered['artifact_id'].tolist())
        record=filtered[filtered['artifact_id']==aid].iloc[0]
        feat=features.get(aid,{})
        art=artifacts.get(aid,{})
        k=feat.get('kernel_features',{})
        st.markdown(f'### {record["checker_verdict"]}')
        st.write(str(record.get('checker_detail','')))
        x,y,z=st.columns(3)
        x.metric('Zero shared stores',k.get('zero_shared_store_count','—'))
        y.metric('CTA memory barriers',k.get('membar_cta_count','—'))
        z.metric('BAR.SYNC instructions',k.get('bar_sync_count','—'))
        st.caption(f'EXIT count: {k.get("exit_count","—")} · Branch count: {k.get("branch_count","—")} · Parser warnings: {len(k.get("parser_warnings",[]))}')
        source=ROOT/'kernels'/'generated'/f'{str(record["zeroization_construct"]).lower()}_{str(record["kernel_context"]).lower()}.cu'
        if source.is_file():
            with st.expander('View CUDA source for this construct and context'):
                st.code(source.read_text(encoding='utf-8',errors='replace'),language='cpp')
                st.caption('Source template is shared across configurations; compiled output depends on build settings.')
        with st.expander('Artifact provenance and SHA-256 records'):
            st.json({key:art.get(key) for key in ('artifact_id','study_phase','compile_status','cubin_hash_sha256','source_hash_sha256','target_arch','cuda_version','ptxas_optimization','kernel_demangled_name')})
            st.caption('Recorded hashes are provenance metadata; they have not been re-verified against the original binaries in this dashboard.')

with tabs[2]:
    st.subheader('Independent checks and recorded evidence')
    cross=js('cross_tool_summary.json')
    repro=js('reproducibility_summary.json')
    p,q,r=st.columns(3)
    p.metric('Cross-tool agreement',f'{cross.get("category_agreement_pct",0):g}%')
    q.metric('Compared artifacts',f'{cross.get("total_artifacts",0):,}')
    r.metric('Disagreements',cross.get('disagreement_count','—'))
    st.caption('Agreement between two disassembly-analysis pipelines; not independent proof of secure runtime erasure.')
    st.divider()
    st.markdown('#### Repeat-build reproducibility')
    a,b,c=st.columns(3)
    a.metric('Repeated compilations',repro.get('n_total_repeat_compiles','—'))
    b.metric('Configurations',repro.get('n_configs','—'))
    c.metric('Category differences','Yes' if repro.get('any_category_difference') else 'No')
    st.json(repro)
    st.markdown('#### Runtime sanity observations')
    runtime=csv('runtime_sanity_summary.csv')
    st.dataframe(runtime,hide_index=True,use_container_width=True)
    st.caption('Recorded runtime checks are separate from static signature classifications; their scope is limited to the tested cases.')
    if (RESULTS/'phase6_structural_reliability.json').exists():
        with st.expander('Structural reliability evidence'):
            st.json(js('phase6_structural_reliability.json'))
    st.warning('The original cubin/SASS paths in artifact metadata refer to a prior environment. Missing binaries are not presented as locally verified.')

with tabs[3]:
    st.subheader('Live verification')
    st.write('Run the existing automated Python test suite or classify a bundled SASS fixture using the project’s real classifier.')
    st.markdown('#### Run pytest')
    if st.button('Run automated tests',type='primary'):
        with st.spinner('Running tests...'):
            try:
                result=subprocess.run([sys.executable,'-m','pytest','-v'],cwd=ROOT,capture_output=True,text=True,timeout=120)
                (st.success if result.returncode==0 else st.error)(f'pytest exit code: {result.returncode}')
                st.code((result.stdout+'\n'+result.stderr)[-18000:],language='text')
            except (subprocess.TimeoutExpired,OSError) as exc:
                st.error(str(exc))
    st.divider()
    st.markdown('#### Classify a bundled SASS fixture')
    fixture_dir=ROOT/'tests'/'fixtures'
    fixtures=sorted(fixture_dir.glob('*.sass')) if fixture_dir.exists() else []
    if fixtures:
        chosen=st.selectbox('SASS fixture',fixtures,format_func=lambda p:p.name)
        kernel=st.text_input('Demangled kernel name',value='kernel_z5_k2' if 'z5_k2' in chosen.name else '')
        with st.expander('View SASS fixture'):
            st.code(chosen.read_text(encoding='utf-8',errors='replace')[:16000],language='text')
        if st.button('Run real classifier'):
            if not kernel.strip():
                st.warning('Enter the exact demangled kernel name in the fixture before running.')
            else:
                try:
                    result=subprocess.run([sys.executable,str(ROOT/'scripts'/'classify_binary.py'),str(chosen),kernel.strip()],cwd=ROOT,capture_output=True,text=True,timeout=30)
                    (st.success if result.returncode==0 else st.error)('Classifier completed' if result.returncode==0 else f'Classifier exited {result.returncode}')
                    st.code(result.stdout+'\n'+result.stderr,language='text')
                except (subprocess.TimeoutExpired,OSError) as exc:
                    st.error(str(exc))
    else:
        st.warning('No bundled SASS fixtures found.')
    st.caption('These actions run locally against test fixtures. They do not compile a new GPU kernel or execute a GPU memory-erasure experiment.')
