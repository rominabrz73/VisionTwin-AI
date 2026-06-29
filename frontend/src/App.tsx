import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from 'react';

type ComponentName = 'Roof' | 'Wall A' | 'Wall B' | 'Floor';
type Risk = 'Low' | 'Medium' | 'High' | 'Critical';
type Result = { prediction:'crack'|'no crack'; confidence:number; model_source:'resnet18'|'demo_heuristic'; processed_image:string; evidence_image:string; preprocessing:{ contour_count:number; edge_pixels:number; threshold_mean:number; detected_crack_area_percent:number; visual_evidence_score:number } };
type Inspection = { id:string; component:ComponentName; prediction:Result['prediction']; confidence:number; risk:Risk; action:string; date:string; healthScore:number; model:string; crackArea:number; evidenceScore:number; contours:number; edgePixels:number };

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const components: ComponentName[] = ['Roof','Wall A','Wall B','Floor'];
const formatDate = (date:string) => new Intl.DateTimeFormat('en-GB',{dateStyle:'medium',timeStyle:'short'}).format(new Date(date));
function decision(prediction:Result['prediction'], confidence:number){
  if(prediction==='no crack') return {risk:'Low' as Risk,action:'Continue routine inspection',health:Math.round(80+confidence*20)};
  if(confidence<.7) return {risk:'Medium' as Risk,action:'Schedule inspection within 30 days',health:Math.round(60-confidence*25)};
  if(confidence<=.9) return {risk:'High' as Risk,action:'Manual inspection within 7 days',health:Math.round(52-confidence*35)};
  return {risk:'Critical' as Risk,action:'Immediate engineer inspection required',health:Math.max(5,Math.round(40-confidence*32))};
}

export default function App(){
  const [file,setFile]=useState<File>(); const [preview,setPreview]=useState('');
  const [component,setComponent]=useState<ComponentName>('Wall A'); const [result,setResult]=useState<Result>();
  const [history,setHistory]=useState<Inspection[]>([]); const [busy,setBusy]=useState(false); const [error,setError]=useState('');
  useEffect(()=>()=>{if(preview)URL.revokeObjectURL(preview)},[preview]);
  const latestByComponent=useMemo(()=>Object.fromEntries(components.map(name=>[name,history.find(x=>x.component===name)])) as Record<ComponentName,Inspection|undefined>,[history]);
  const support=result?decision(result.prediction,result.confidence):undefined;
  const assetRecords=components.map(name=>latestByComponent[name]).filter(Boolean) as Inspection[];
  const buildingHealth=assetRecords.length?Math.round(assetRecords.reduce((sum,x)=>sum+x.healthScore,0)/assetRecords.length):null;
  const openIssues=assetRecords.filter(x=>x.prediction==='crack').length;
  const highRisk=assetRecords.filter(x=>x.risk==='High'||x.risk==='Critical').length;

  function chooseFile(event:ChangeEvent<HTMLInputElement>){const next=event.target.files?.[0];if(preview)URL.revokeObjectURL(preview);setFile(next);setPreview(next?URL.createObjectURL(next):'');setResult(undefined);setError('')}
  async function inspect(event:FormEvent){event.preventDefault();if(!file)return;setBusy(true);setError('');try{const body=new FormData();body.append('file',file);const response=await fetch(`${API_URL}/predict`,{method:'POST',body});if(!response.ok)throw new Error((await response.json()).detail??'Inspection failed');const next:Result=await response.json();const nextSupport=decision(next.prediction,next.confidence);const record:Inspection={id:crypto.randomUUID(),component,prediction:next.prediction,confidence:next.confidence,risk:nextSupport.risk,action:nextSupport.action,date:new Date().toISOString(),healthScore:nextSupport.health,model:next.model_source==='resnet18'?'ResNet18':'Demo heuristic',crackArea:next.preprocessing.detected_crack_area_percent,evidenceScore:next.preprocessing.visual_evidence_score,contours:next.preprocessing.contour_count,edgePixels:next.preprocessing.edge_pixels};setResult(next);setHistory(previous=>[record,...previous].slice(0,20))}catch(e){setError(e instanceof Error?e.message:'Inspection failed')}finally{setBusy(false)}}
  function generateReport(){if(!result||!support)return;const date=history[0]?.date??new Date().toISOString();const conclusion=result.prediction==='crack'?`${support.risk} risk visual anomaly identified. ${support.action}.`:'No crack was classified. Continue the routine inspection programme.';const html=`<!doctype html><html><head><meta charset="utf-8"><title>VisionTwin AI Report</title><style>body{font:15px Arial;max-width:760px;margin:50px auto;color:#17233b}h1{border-bottom:3px solid #ef4f41;padding-bottom:14px}table{width:100%;border-collapse:collapse}td{padding:11px;border-bottom:1px solid #ddd}td:first-child{font-weight:bold}.notice{background:#f5f7fb;padding:18px;border-left:4px solid #ef4f41}</style></head><body><h1>VisionTwin AI Inspection Report</h1><table><tr><td>Date</td><td>${formatDate(date)}</td></tr><tr><td>Component</td><td>${component}</td></tr><tr><td>Prediction</td><td>${result.prediction.toUpperCase()}</td></tr><tr><td>Confidence</td><td>${Math.round(result.confidence*100)}%</td></tr><tr><td>Risk</td><td>${support.risk}</td></tr><tr><td>Action</td><td>${support.action}</td></tr><tr><td>Model</td><td>ResNet18 v1.0.0</td></tr><tr><td>Evidence</td><td>${result.preprocessing.detected_crack_area_percent}% detected area; ${result.preprocessing.contour_count} contours</td></tr><tr><td>Conclusion</td><td>${conclusion}</td></tr></table><p class="notice">This prototype supports inspection prioritisation using AI and Computer Vision. It is designed to assist engineers rather than replace professional engineering judgement.</p></body></html>`;const url=URL.createObjectURL(new Blob([html],{type:'text/html'}));const a=document.createElement('a');a.href=url;a.download='visiontwin-inspection-report.html';a.click();URL.revokeObjectURL(url)}
  const assetClass=(name:ComponentName)=>latestByComponent[name]?(latestByComponent[name]?.prediction==='crack'?'asset-alert':'asset-clear'):'';

  return <div className="app-shell">
    <aside className="side-nav">
      <div className="brand"><span>◆</span><div><h1>VisionTwin <b>AI</b></h1><p>Infrastructure Intelligence</p></div></div>
      <nav><a className="active" href="#dashboard"><span>⌂</span>Dashboard</a><a href="#new-inspection"><span>▣</span>New Inspection</a><a href="#history"><span>◷</span>Inspection History</a><a href="#report"><span>▤</span>Reports</a><a href="#twin"><span>▥</span>Assets</a></nav>
      <div className="model-info"><h3>Model Information</h3><dl><div><dt>Model</dt><dd>ResNet18</dd></div><div><dt>Version</dt><dd>v1.0.0</dd></div><div><dt>Framework</dt><dd>PyTorch</dd></div><div><dt>Inference</dt><dd>Local</dd></div></dl></div>
    </aside>
    <main className="dashboard-main" id="dashboard">
      <header className="topbar"><button className="menu-button" aria-label="Toggle navigation">☰</button><div><span className="model-pill">ResNet18 · v1.0.0</span><span className="online"><i/>System online</span></div></header>
      <div className="dashboard-content">
        <section className="kpi-grid">
          <Kpi icon="♡" label="Building Health" value={buildingHealth===null?'—':`${buildingHealth}%`} tone="green" note={buildingHealth===null?'Awaiting inspection':'Latest asset average'}/>
          <Kpi icon="▣" label="Total Inspections" value={history.length} tone="blue" note="Current session"/>
          <Kpi icon="△" label="Open Issues" value={openIssues} tone="orange" note={openIssues?'Needs attention':'No open issues'}/>
          <Kpi icon="◇" label="High Risk Areas" value={highRisk} tone="red" note={highRisk?'Requires action':'None identified'}/>
          <Kpi icon="□" label="Last Inspection" value={history[0]?'Today':'—'} tone="blue" note={history[0]?formatDate(history[0].date):'Not recorded'}/>
        </section>

        <section className="dashboard-grid">
          <div className="primary-column">
            <section className="dash-card inspection-card" id="new-inspection">
              <div className="card-heading"><div><h2>{result?'Latest Inspection Result':'New Inspection'}</h2><p>{result?'AI classification with supporting visual evidence.':'Select an asset and upload a close surface image.'}</p></div>{result&&<span className={`risk-badge ${support?.risk.toLowerCase()}`}>{support?.risk} risk</span>}</div>
              <form className="compact-form" onSubmit={inspect}><label>Asset component<select value={component} onChange={e=>setComponent(e.target.value as ComponentName)}>{components.map(x=><option key={x}>{x}</option>)}</select></label><label className="file-control"><input type="file" accept="image/jpeg,image/png,image/webp" onChange={chooseFile}/><span>{file?file.name:'Choose inspection image'}</span></label><button disabled={!file||busy}>{busy?'Analysing…':'Run Inspection'}</button></form>{error&&<p className="dash-error">{error}</p>}
              {result?<><div className="result-layout"><div className="image-comparison"><ImageTile label="Original Image" src={preview}/><ImageTile label="Detection Overlay" src={result.evidence_image}/></div><div className="result-metrics"><Metric label="Prediction" value={result.prediction==='crack'?'Crack Detected':'No Crack Detected'} tone={result.prediction==='crack'?'red':'green'}/><Metric label="Confidence" value={`${(result.confidence*100).toFixed(1)}%`} progress={result.confidence}/><Metric label="Risk Level" value={support?.risk??'—'} tone={support?.risk.toLowerCase()}/><Metric label="Recommended Action" value={support?.action??'—'} tone={result.prediction==='crack'?'red':'green'}/></div></div><h3 className="evidence-title">Image Processing Evidence</h3><div className="evidence-thumbs"><ImageTile label="Processed Edge Map" src={result.processed_image}/><ImageTile label="Contour Overlay" src={result.evidence_image}/><EvidenceStat label="Detected Area" value={`${result.preprocessing.detected_crack_area_percent}%`}/><EvidenceStat label="Contours" value={result.preprocessing.contour_count}/><EvidenceStat label="Visual Score" value={`${result.preprocessing.visual_evidence_score}/100`}/></div></>:<div className="upload-empty"><label><input type="file" accept="image/jpeg,image/png,image/webp" onChange={chooseFile}/><span>↑</span><strong>Upload an infrastructure image</strong><p>JPEG, PNG or WebP · close, well-lit images work best</p></label></div>}
            </section>

            <section className="dash-card history-table" id="history"><div className="card-heading"><div><h2>Inspection History</h2><p>Results completed during this session.</p></div>{history.length>0&&<button className="clear-btn" onClick={()=>{setHistory([]);setResult(undefined)}}>Clear</button>}</div>{history.length?<div className="table-wrap"><table><thead><tr><th>Date</th><th>Asset</th><th>Prediction</th><th>Confidence</th><th>Risk</th><th>Recommended Action</th></tr></thead><tbody>{history.map(item=><tr key={item.id}><td>{formatDate(item.date)}</td><td><b>{item.component}</b></td><td className={item.prediction==='crack'?'cell-red':'cell-green'}>● {item.prediction==='crack'?'Crack Detected':'No Crack'}</td><td>{(item.confidence*100).toFixed(1)}%</td><td><span className={`risk-badge ${item.risk.toLowerCase()}`}>{item.risk}</span></td><td>{item.action}</td></tr>)}</tbody></table></div>:<div className="table-empty">No inspections yet. Upload an image to create the first record.</div>}</section>
          </div>

          <aside className="right-rail">
            <section className="dash-card twin-card" id="twin"><div className="card-heading"><div><h2>Asset Condition Twin</h2><p>Component-level condition state</p></div></div><div className="mini-building"><div className={`mini-roof ${assetClass('Roof')}`}>ROOF<small>{latestByComponent.Roof?.prediction==='crack'?'Issue':latestByComponent.Roof?'Clear':'Not inspected'}</small></div><div className="mini-walls"><div className={assetClass('Wall A')}>WALL A<small>{latestByComponent['Wall A']?.prediction==='crack'?'Issue':latestByComponent['Wall A']?'Clear':'Not inspected'}</small></div><div className={assetClass('Wall B')}>WALL B<small>{latestByComponent['Wall B']?.prediction==='crack'?'Issue':latestByComponent['Wall B']?'Clear':'Not inspected'}</small></div></div><div className={`mini-floor ${assetClass('Floor')}`}>FLOOR</div></div><div className="twin-legend"><span><i/>Not inspected</span><span><i className="green"/>Clear</span><span><i className="red"/>Issue</span></div></section>
            <section className="dash-card summary-card"><div className="card-heading"><div><h2>Inspection Summary</h2></div></div><dl><div><dt>Asset Component</dt><dd>{result?component:'—'}</dd></div><div><dt>Inspection Date</dt><dd>{history[0]?formatDate(history[0].date):'—'}</dd></div><div><dt>Model</dt><dd>ResNet18</dd></div><div><dt>Model Version</dt><dd>v1.0.0</dd></div></dl><button id="report" disabled={!result} onClick={generateReport}>▤ Generate Inspection Report</button></section>
            <section className="dash-card risk-guide"><div className="card-heading"><div><h2>Risk Level Guide</h2></div></div><Guide tone="low" label="Low" text="Continue routine inspection"/><Guide tone="medium" label="Medium" text="Inspect within 30 days"/><Guide tone="high" label="High" text="Manual inspection within 7 days"/><Guide tone="critical" label="Critical" text="Immediate engineer inspection"/></section>
          </aside>
        </section>
        <footer>This prototype supports inspection prioritisation and does not replace professional engineering judgement.</footer>
      </div>
    </main>
  </div>
}

function Kpi({icon,label,value,tone,note}:{icon:string;label:string;value:string|number;tone:string;note:string}){return <article className="kpi-card"><span className={`kpi-icon ${tone}`}>{icon}</span><div><small>{label}</small><strong className={tone}>{value}</strong><p>{note}</p></div></article>}
function ImageTile({label,src}:{label:string;src:string}){return <figure className="image-tile"><figcaption>{label}</figcaption><img src={src} alt={label}/></figure>}
function Metric({label,value,tone='',progress}:{label:string;value:string;tone?:string;progress?:number}){return <div className="metric-box"><small>{label}</small><strong className={tone}>{value}</strong>{progress!==undefined&&<div className="progress"><i style={{width:`${progress*100}%`}}/></div>}</div>}
function EvidenceStat({label,value}:{label:string;value:string|number}){return <div className="evidence-stat"><small>{label}</small><strong>{value}</strong></div>}
function Guide({tone,label,text}:{tone:string;label:string;text:string}){return <div className="guide-row"><i className={tone}/><div><strong>{label}</strong><p>{text}</p></div></div>}
