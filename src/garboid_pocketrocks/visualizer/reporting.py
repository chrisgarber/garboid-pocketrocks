# ruff: noqa: E501
"""Render a self-contained interactive tournament insight report."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def render_insights_html(
    tournament: dict[str, Any],
    bots: dict[str, dict[str, Any]],
) -> str:
    payload = json.dumps(
        {"tournament": tournament, "bots": bots},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).replace("</", "<\\/")
    return _REPORT_TEMPLATE.replace("__INSIGHT_PAYLOAD__", payload)


def write_insights_html(path: Path, content: str, *, overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"visualizer output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


_REPORT_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PocketRocks tournament insights</title>
<style>
:root{color-scheme:light dark;--bg:#f7f6f2;--panel:#fff;--ink:#18211d;--muted:#657069;--line:#d9ded9;--accent:#16745a;--accent2:#c77b30;--bad:#ad4538;--soft:#e5f1ec;--shadow:0 16px 45px rgba(24,33,29,.08)}
@media(prefers-color-scheme:dark){:root{--bg:#111612;--panel:#19201b;--ink:#edf3ef;--muted:#a9b4ad;--line:#364038;--accent:#6bc5a4;--accent2:#e9aa68;--bad:#ef8d80;--soft:#243d33;--shadow:none}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,select{font:inherit}button:focus-visible,select:focus-visible{outline:3px solid color-mix(in srgb,var(--accent) 55%,transparent);outline-offset:2px}.shell{max-width:1240px;margin:auto;padding:32px 24px 64px}header{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:24px}h1{font-family:ui-serif,Georgia,serif;font-size:clamp(2rem,4vw,3.5rem);line-height:1;margin:0;letter-spacing:-.035em}h2{font-size:1.1rem;margin:0 0 18px}h3{font-size:.95rem;margin:0}.eyebrow{text-transform:uppercase;letter-spacing:.13em;color:var(--accent);font-size:.75rem;font-weight:700;margin-bottom:9px}.meta,.muted{color:var(--muted)}.meta{margin-top:10px}.tabs{display:flex;gap:6px;padding:5px;border:1px solid var(--line);border-radius:14px;background:var(--panel)}.tabs button{border:0;border-radius:9px;background:transparent;color:var(--muted);padding:9px 13px;cursor:pointer}.tabs button[aria-selected="true"]{background:var(--ink);color:var(--bg)}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:18px}.toolbar label{font-size:.85rem;color:var(--muted)}select{min-width:260px;margin-left:8px;padding:9px 32px 9px 10px;border:1px solid var(--line);border-radius:9px;background:var(--panel);color:var(--ink)}.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:16px}.panel,.stat{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}.panel{grid-column:span 6;padding:20px;min-width:0}.panel.wide{grid-column:1/-1}.stat{grid-column:span 3;padding:17px}.stat .value{font:600 1.65rem/1.1 ui-serif,Georgia,serif;margin-top:7px}.stat .label{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.08em}.chart{width:100%;min-height:190px}.chart svg{display:block;width:100%;height:auto;overflow:visible}.axis{stroke:var(--line);stroke-width:1}.gridline{stroke:var(--line);stroke-width:1}.mark{fill:var(--accent)}.interval{stroke:var(--accent);stroke-width:2}.reference{stroke:var(--muted);stroke-dasharray:5 5}.chart text{fill:var(--muted);font-size:11px}.chart .primary-label{fill:var(--ink);font-size:12px}.series{fill:none;stroke:var(--accent);stroke-width:2}.series.secondary{stroke:var(--accent2)}.empty{display:grid;place-items:center;min-height:190px;border:1px dashed var(--line);border-radius:12px;color:var(--muted);text-align:center;padding:24px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:.84rem}th{text-align:left;color:var(--muted);font-weight:600}th,td{padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.heat td{min-width:66px;text-align:center}.heat .cell{background:color-mix(in srgb,var(--accent) calc(var(--strength)*70%),transparent);color:var(--ink);border-radius:7px;padding:7px;display:block;font-variant-numeric:tabular-nums}.legend{display:flex;gap:14px;align-items:center;color:var(--muted);font-size:.78rem;margin-top:10px}.swatch{width:22px;height:3px;background:var(--accent);display:inline-block}.swatch.secondary{background:var(--accent2)}.note{font-size:.78rem;color:var(--muted);margin:10px 0 0}.hidden{display:none!important}@media(max-width:820px){header{align-items:start;flex-direction:column}.panel{grid-column:1/-1}.stat{grid-column:span 6}.toolbar{align-items:start;flex-direction:column}select{min-width:min(100%,320px);margin:6px 0 0}}@media(max-width:480px){.shell{padding:22px 14px 42px}.stat{grid-column:1/-1}.tabs{width:100%}.tabs button{flex:1}.panel{padding:16px}}
</style>
</head>
<body>
<div class="shell">
<header><div><div class="eyebrow">Garboid · diagnostic report</div><h1>Tournament insights</h1><div id="meta" class="meta"></div></div><div class="tabs" role="tablist" aria-label="Insight engine"><button id="tab-tournament" role="tab" aria-selected="true">Tournament</button><button id="tab-bot" role="tab" aria-selected="false">Bot deep dive</button></div></header>
<main>
<section id="tournament-view" role="tabpanel" aria-labelledby="tab-tournament">
<div id="tournament-stats" class="grid"></div>
<div class="grid" style="margin-top:16px">
<section class="panel wide"><h2>Rating and uncertainty</h2><div id="ratings" class="chart"></div></section>
<section class="panel wide"><h2>Head-to-head score</h2><div id="matchups"></div><p class="note">Each cell is the row bot's pairwise score when both bots appeared: win 1, tie ½, loss 0.</p></section>
<section class="panel"><h2>Outright win rate by value chart</h2><div id="field-charts"></div></section>
<section class="panel"><h2>Model calibration</h2><div id="calibration" class="chart"></div></section>
<section class="panel wide"><h2>Leaderboard detail</h2><div id="leaderboard"></div></section>
</div></section>
<section id="bot-view" class="hidden" role="tabpanel" aria-labelledby="tab-bot">
<div class="toolbar"><div><div class="eyebrow">Single-bot engine</div><h2 id="bot-heading">Bot deep dive</h2></div><label for="bot-select">Bot<select id="bot-select"></select></label></div>
<div id="bot-stats" class="grid"></div>
<div class="grid" style="margin-top:16px">
<section class="panel"><h2>Performance against opponents</h2><div id="bot-opponents" class="chart"></div></section>
<section class="panel"><h2>Games with an objective claim</h2><div id="bot-objectives" class="chart"></div><p class="note">Wilson 95% intervals; the rate is conditioned on each opponent being present.</p></section>
<section class="panel"><h2>Profit on won resource auctions</h2><div id="bot-profit" class="chart"></div><p class="note">Terminal resource value under that game's chart minus the winning payment.</p></section>
<section class="panel"><h2>Winning prices for investments</h2><div id="bot-investments" class="chart"></div><p class="note">The payment is locked principal returned at scoring, not a cost. Net profit is always $5 for Invest $5 and $10 for Invest $10.</p></section>
<section class="panel"><h2>Loan valuation</h2><div id="bot-loans" class="chart"></div><p class="note">Winning-price distributions; up-front liquidity is principal minus payment, not accounting profit.</p></section>
<section class="panel"><h2>Outright wins by value chart</h2><div id="bot-charts" class="chart"></div></section>
<section class="panel"><h2>Cash-starved requests by turn</h2><div id="bot-cash" class="chart"></div><div class="legend"><span><i class="swatch"></i> zero cash</span><span><i class="swatch secondary"></i> legally unable to bid</span></div></section>
<section class="panel"><h2>Acquisition mix</h2><div id="bot-actions"></div></section>
<section class="panel"><h2>Terminal score composition</h2><div id="bot-score"></div></section>
<section class="panel wide"><h2>Bidding and liquidity by action</h2><div id="bot-bidding"></div></section>
</div></section>
</main></div>
<script id="insight-data" type="application/json">__INSIGHT_PAYLOAD__</script>
<script>
const data=JSON.parse(document.getElementById('insight-data').textContent);const NS='http://www.w3.org/2000/svg';
const $=id=>document.getElementById(id),pct=v=>`${(100*v).toFixed(1)}%`,money=v=>`$${Number(v).toFixed(1)}`,num=v=>Number(v).toLocaleString();
const esc=value=>String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
function empty(id,message){$(id).innerHTML=`<div class="empty">${message}</div>`}
function svg(width,height,label){const s=document.createElementNS(NS,'svg');s.setAttribute('viewBox',`0 0 ${width} ${height}`);s.setAttribute('role','img');s.setAttribute('aria-label',label);return s}
function node(name,attrs={},text=''){const n=document.createElementNS(NS,name);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text)n.textContent=text;return n}
function stat(label,value,context=''){return `<article class="stat"><div class="label">${label}</div><div class="value">${value}</div>${context?`<div class="note">${context}</div>`:''}</article>`}
function forest(id,rows,{label='opponent_name',value='score',low='lower',high='upper',format=pct,reference=.5,domain=[0,1],emptyText='No game-level evidence was captured.'}={}){if(!rows.length){empty(id,emptyText);return}const w=760,rowH=30,h=34+rows.length*rowH,maxLabel=220,left=maxLabel+10,right=55,scale=x=>left+(x-domain[0])/(domain[1]-domain[0])*(w-left-right);const s=svg(w,h,'Point estimates with 95 percent intervals');if(reference!==null)s.append(node('line',{x1:scale(reference),x2:scale(reference),y1:8,y2:h-25,class:'reference'}));rows.forEach((r,i)=>{const y=18+i*rowH;s.append(node('text',{x:0,y:y+4,class:'primary-label'},String(r[label]).slice(0,31)));s.append(node('line',{x1:scale(r[low]),x2:scale(r[high]),y1:y,y2:y,class:'interval'}));s.append(node('circle',{cx:scale(r[value]),cy:y,r:5,class:'mark'}));s.append(node('text',{x:w-right+8,y:y+4},format(r[value])));const title=node('title',{},`${r[label]}: ${format(r[value])}, interval ${format(r[low])}–${format(r[high])}, ${num(r.games)} games`);s.lastChild.append(title)});s.append(node('line',{x1:left,x2:w-right,y1:h-22,y2:h-22,class:'axis'}));[domain[0],(domain[0]+domain[1])/2,domain[1]].forEach(v=>s.append(node('text',{x:scale(v),y:h-5,'text-anchor':'middle'},format(v))));$(id).replaceChildren(s)}
function ratingChart(){const rows=data.tournament.leaderboard;if(!rows.length){empty('ratings','No ratings.');return}const lows=rows.map(r=>r.rating_interval_lower??r.pl_rating),highs=rows.map(r=>r.rating_interval_upper??r.pl_rating),lo=Math.floor((Math.min(...lows)-25)/50)*50,hi=Math.ceil((Math.max(...highs)+25)/50)*50,w=900,rowH=30,h=38+rows.length*rowH,left=245,right=62,scale=x=>left+(x-lo)/(hi-lo)*(w-left-right),s=svg(w,h,'Plackett-Luce ratings with bootstrap intervals');rows.forEach((r,i)=>{const y=18+i*rowH;s.append(node('text',{x:0,y:y+4,class:'primary-label'},`${r.rank}. ${r.bot_name}`));s.append(node('line',{x1:scale(r.rating_interval_lower??r.pl_rating),x2:scale(r.rating_interval_upper??r.pl_rating),y1:y,y2:y,class:'interval'}));s.append(node('circle',{cx:scale(r.pl_rating),cy:y,r:5,class:'mark'}));s.append(node('text',{x:w-right+7,y:y+4},Math.round(r.pl_rating)))});s.append(node('line',{x1:left,x2:w-right,y1:h-23,y2:h-23,class:'axis'}));[lo,(lo+hi)/2,hi].forEach(v=>s.append(node('text',{x:scale(v),y:h-5,'text-anchor':'middle'},Math.round(v))));$('ratings').replaceChildren(s)}
function matchupTable(){const rows=data.tournament.leaderboard,values=data.tournament.matchups;if(!values.length){empty('matchups','Run the tournament with --decision-reports to unlock opponent matchups.');return}const by=new Map(values.map(r=>[`${r.bot_id}|${r.opponent_id}`,r]));let html='<div class="table-wrap"><table class="heat"><thead><tr><th>Bot</th>'+rows.map(r=>`<th title="${esc(r.bot_name)}">${esc(r.bot_name.slice(0,12))}</th>`).join('')+'</tr></thead><tbody>';for(const a of rows){html+=`<tr><th>${esc(a.bot_name)}</th>`;for(const b of rows){if(a.bot_id===b.bot_id){html+='<td>—</td>';continue}const r=by.get(`${a.bot_id}|${b.bot_id}`);html+=r?`<td><span class="cell" style="--strength:${r.score.toFixed(3)}" title="${num(r.games)} games; 95% interval ${pct(r.lower)}–${pct(r.upper)}">${pct(r.score)}</span></td>`:'<td>n/a</td>'}html+='</tr>'}html+='</tbody></table></div>';$('matchups').innerHTML=html}
function fieldCharts(){const bots=data.tournament.leaderboard,rows=data.tournament.conditions,charts=[...new Set(rows.map(r=>r.chart))].sort();if(!rows.length){empty('field-charts','No condition statistics.');return}const agg=new Map;for(const r of rows){const k=`${r.bot_id}|${r.chart}`,a=agg.get(k)||{wins:0,games:0};a.wins+=r.outright_wins;a.games+=r.games;agg.set(k,a)}let html='<div class="table-wrap"><table class="heat"><thead><tr><th>Bot</th>'+charts.map(c=>`<th>${esc(c)}</th>`).join('')+'</tr></thead><tbody>';for(const b of bots){html+=`<tr><th>${esc(b.bot_name)}</th>`;for(const c of charts){const a=agg.get(`${b.bot_id}|${c}`),v=a?a.wins/a.games:0;html+=`<td><span class="cell" style="--strength:${v.toFixed(3)}" title="${a?num(a.games):0} games">${pct(v)}</span></td>`}html+='</tr>'}html+='</tbody></table></div>';$('field-charts').innerHTML=html}
function linePlot(id,rows,{x,y,secondary=null,xFormat=String,yFormat=pct,label='Line chart',xDomain=null,yCeiling=null}={}){if(!rows.length){empty(id,'No decision traces were captured.');return}const w=760,h=250,left=48,right=24,top=15,bottom=34,xVals=rows.map(x),maxX=xDomain?xDomain[1]:Math.max(...xVals),minX=xDomain?xDomain[0]:Math.min(...xVals),values=rows.flatMap(r=>secondary?[y(r),secondary(r)]:[y(r)]),maxY=yCeiling??Math.max(.01,...values),sx=v=>left+(v-minX)/(Math.max(1e-9,maxX-minX))*(w-left-right),sy=v=>top+(1-v/maxY)*(h-top-bottom),s=svg(w,h,label);[0,.5,1].forEach(f=>{const yy=top+f*(h-top-bottom);s.append(node('line',{x1:left,x2:w-right,y1:yy,y2:yy,class:'gridline'}));s.append(node('text',{x:left-7,y:yy+4,'text-anchor':'end'},yFormat(maxY*(1-f))))});function path(fn,cls){const d=rows.map((r,i)=>`${i?'L':'M'}${sx(x(r)).toFixed(1)},${sy(fn(r)).toFixed(1)}`).join(' ');s.append(node('path',{d,class:cls}))}path(y,'series');if(secondary)path(secondary,'series secondary');[minX,maxX].forEach(v=>s.append(node('text',{x:sx(v),y:h-8,'text-anchor':'middle'},xFormat(v))));$(id).replaceChildren(s)}
function calibration(){const rows=data.tournament.calibration;if(!rows.length){empty('calibration','No calibration bins.');return}linePlot('calibration',rows,{x:r=>r.mean_prediction,y:r=>r.observed_score,xFormat:pct,yFormat:pct,label:'Predicted versus observed pairwise scores',xDomain:[0,1],yCeiling:1});const s=$('calibration').querySelector('svg'),w=760,h=250,left=48,right=24,top=15,bottom=34,scaleX=v=>left+v*(w-left-right),scaleY=v=>top+(1-v)*(h-top-bottom);s.prepend(node('line',{x1:scaleX(0),x2:scaleX(1),y1:scaleY(0),y2:scaleY(1),class:'reference'}))}
function leaderboard(){const rows=data.tournament.leaderboard;let html='<div class="table-wrap"><table><thead><tr><th>Rank</th><th>Bot</th><th class="num">Rating</th><th class="num">Games</th><th class="num">Outright wins</th><th class="num">Mean finish</th><th class="num">Mean money</th><th class="num">Faults</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${r.rank}</td><td>${esc(r.bot_name)}</td><td class="num">${Math.round(r.pl_rating)}</td><td class="num">${num(r.games)}</td><td class="num">${num(r.outright_wins)}</td><td class="num">${pct(r.mean_normalized_finish)}</td><td class="num">${money(r.mean_final_money)}</td><td class="num">${num(r.faults)}</td></tr>`).join('')+'</tbody></table></div>';$('leaderboard').innerHTML=html}
function boxPlot(id,rows,{label='action',format=money,emptyText='No resolved game detail was captured.'}={}){if(!rows.length){empty(id,emptyText);return}const all=rows.flatMap(r=>[r.p10,r.p25,r.median,r.p75,r.p90]),lo=Math.min(0,...all),hi=Math.max(1,...all),w=760,h=45+rows.length*42,left=210,right=55,scale=x=>left+(x-lo)/(hi-lo)*(w-left-right),s=svg(w,h,'Distribution from tenth to ninetieth percentile');if(lo<0&&hi>0)s.append(node('line',{x1:scale(0),x2:scale(0),y1:7,y2:h-27,class:'reference'}));rows.forEach((r,i)=>{const y=22+i*42;s.append(node('text',{x:0,y:y+4,class:'primary-label'},r[label]));s.append(node('line',{x1:scale(r.p10),x2:scale(r.p90),y1:y,y2:y,class:'interval'}));s.append(node('rect',{x:scale(r.p25),y:y-7,width:Math.max(2,scale(r.p75)-scale(r.p25)),height:14,fill:'var(--soft)',stroke:'var(--accent)'}));s.append(node('line',{x1:scale(r.median),x2:scale(r.median),y1:y-8,y2:y+8,stroke:'var(--accent)','stroke-width':2}));s.append(node('text',{x:w-right+6,y:y+4},format(r.median)));s.lastChild.append(node('title',{},`n=${num(r.count)}, mean ${format(r.mean)}, median ${format(r.median)}`))});s.append(node('line',{x1:left,x2:w-right,y1:h-24,y2:h-24,class:'axis'}));[lo,(lo+hi)/2,hi].forEach(v=>s.append(node('text',{x:scale(v),y:h-6,'text-anchor':'middle'},format(v))));$(id).replaceChildren(s)}
function aggregateCharts(rows){const map=new Map;for(const r of rows){const a=map.get(r.chart)||{chart:r.chart,games:0,wins:0};a.games+=r.games;a.wins+=r.win_rate*r.games;map.set(r.chart,a)}return [...map.values()].map(a=>{const p=a.wins/a.games,z=1.959963984540054,d=1+z*z/a.games,c=(p+z*z/(2*a.games))/d,s=z*Math.sqrt(p*(1-p)/a.games+z*z/(4*a.games*a.games))/d;return{chart:a.chart,games:a.games,score:p,lower:Math.max(0,c-s),upper:Math.min(1,c+s)}})}
function simpleTable(id,headers,rows){if(!rows.length){empty(id,'No evidence captured.');return}$(id).innerHTML='<div class="table-wrap"><table><thead><tr>'+headers.map(h=>`<th class="${h.num?'num':''}">${h.label}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+headers.map(h=>`<td class="${h.num?'num':''}">${h.format?h.format(r[h.key]):r[h.key]}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>'}
function updateBot(id){const b=data.bots[id],s=b.summary;$('bot-heading').textContent=b.bot_name;const phases=b.cash_by_phase,all=phases.reduce((a,r)=>({requests:a.requests+r.requests,zero:a.zero+r.cash_zero_rate*r.requests,hard:a.hard+r.hard_constrained_rate*r.requests}),{requests:0,zero:0,hard:0});$('bot-stats').innerHTML=stat('PL rating',Math.round(s.pl_rating),`rank ${s.rank} of ${Object.keys(data.bots).length}`)+stat('Outright win rate',pct(s.outright_wins/s.games),`${num(s.games)} games`)+stat('Mean final money',money(s.mean_final_money),`mean finish ${pct(s.mean_normalized_finish)}`)+stat('Cash-zero requests',all.requests?pct(all.zero/all.requests):'n/a',all.requests?`${pct(all.hard/all.requests)} legally unable to bid`:'requires decision traces');forest('bot-opponents',b.opponents);forest('bot-objectives',b.objectives,{value:'game_rate'});boxPlot('bot-profit',b.auction_profit);boxPlot('bot-investments',b.investment_prices);boxPlot('bot-loans',b.loans);forest('bot-charts',aggregateCharts(b.value_charts),{label:'chart'});linePlot('bot-cash',b.cash_by_turn,{x:r=>r.turn,y:r=>r.cash_zero_rate,secondary:r=>r.hard_constrained_rate,xFormat:v=>`turn ${v}`,yFormat:pct,label:'Cash pressure by action-deck turn'});simpleTable('bot-actions',[{key:'action',label:'Action'},{key:'wins',label:'Wins',num:true,format:num},{key:'per_100_games',label:'Per 100 games',num:true,format:v=>Number(v).toFixed(1)}],b.action_wins);simpleTable('bot-score',[{key:'component',label:'Component'},{key:'mean',label:'Mean',num:true,format:money},{key:'median',label:'Median',num:true,format:money}],b.score_components);simpleTable('bot-bidding',[{key:'action',label:'Action'},{key:'requests',label:'Requests',num:true,format:num},{key:'mean_bid',label:'Mean bid',num:true,format:money},{key:'mean_cash',label:'Mean cash',num:true,format:money},{key:'pass_rate',label:'Pass',num:true,format:pct},{key:'cash_zero_rate',label:'Zero cash',num:true,format:pct},{key:'hard_constrained_rate',label:'Hard constrained',num:true,format:pct},{key:'cap_binding_rate',label:'At bid cap',num:true,format:pct}],b.bidding_by_action)}
function setView(view){const tournament=view==='tournament';$('tournament-view').classList.toggle('hidden',!tournament);$('bot-view').classList.toggle('hidden',tournament);$('tab-tournament').setAttribute('aria-selected',String(tournament));$('tab-bot').setAttribute('aria-selected',String(!tournament))}
const cfg=data.tournament.configuration;$('meta').textContent=`${num(cfg.games||0)} games · ${(cfg.bots||[]).length} bots · charts ${(cfg.charts||[]).join(', ')}`;$('tournament-stats').innerHTML=stat('Games',num(cfg.games||0))+stat('Bots',num((cfg.bots||[]).length))+stat('Pair outcomes',num(data.tournament.pair_outcomes||0))+stat('Diagnostic depth',data.tournament.availability.decision_traces?'full':'summary only',data.tournament.availability.decision_traces?'turns, objectives, and liquidity':'rerun with --decision-reports for bot behavior');
ratingChart();matchupTable();fieldCharts();calibration();leaderboard();const select=$('bot-select');Object.values(data.bots).sort((a,b)=>a.summary.rank-b.summary.rank).forEach(b=>select.add(new Option(`${b.summary.rank}. ${b.bot_name}`,b.bot_id)));if(select.value)updateBot(select.value);select.addEventListener('change',e=>updateBot(e.target.value));$('tab-tournament').addEventListener('click',()=>setView('tournament'));$('tab-bot').addEventListener('click',()=>setView('bot'));
</script>
</body></html>
"""
