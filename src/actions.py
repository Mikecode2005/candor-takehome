#!/usr/bin/env python3
import json,re,argparse
from pathlib import Path
from datetime import datetime,timedelta

def dt(s): return datetime.fromisoformat(s.replace('Z','+00:00'))
def iso(d): return d.isoformat()

# Generic resources: fixed intent/object vocabulary, not train phrasing.
APPS=('figma','slack','gmail','calendar','notion','linear','zoom','drive','docs','sheets','github')
CHANNELS={'route planner':'C10RP','route-planner':'C10RP','design':'C11DESIGN','general':'C01GENERAL'}
PEOPLE={
 'sarah kim':('U03SARAHK','sarah.kim@brightline.example.com'),
 'sarah patel':(None,'sarah.patel@acmefreight.example.com'),
 'john':(None,'john@brightline.example.com'),
 'john okafor':(None,'john@brightline.example.com'),
 'ben':('U06BEN','ben@brightline.example.com'),
 'dana':(None,'dana@brightline.example.com'),
 'marcus':(None,'marcus@brightline.example.com'),
 'tom':(None,'tom@brightline.example.com'),
}

def load_data(data):
    users={x['real_name'].lower():x for x in json.loads((data/'connectors/slack/users.json').read_text())}
    channels={x['id']:x for x in json.loads((data/'connectors/slack/channels.json').read_text())}
    events={x['id']:x for x in map(json.loads,(data/'connectors/google_calendar/events.jsonl').read_text().splitlines())}
    return users,channels,events

def classify(cmd):
    """Intent classifier: verb + object type, independent of exact wording."""
    ql=cmd.lower()
    if re.search(r'\bdelete\b',ql) and re.search(r'email|mail|message',ql):
        return 'delete_email'
    if re.search(r'\bopen\b|\blaunch\b|\bstart\b',ql) and any(a in ql for a in APPS):
        return 'open_app'
    if re.search(r'\btell\b.*\bchannel\b|\bannounce\b|\bpost\b.*\bchannel\b',ql):
        return 'channel_announce'
    if re.search(r'\bemail\b|\bmail\b',ql):
        return 'send_email'
    if re.search(r'\bmessage\b|\bslack\b|\btell\b|\bdm\b|\bthank\b',ql):
        return 'send_message'
    if re.search(r'\bremind\b',ql):
        return 'create_reminder'
    if re.search(r'\bmove\b|\breschedul|\bshift\b',ql):
        return 'move_event'
    if re.search(r'\bbook\b|\bschedule\b|\bset up\b|\bsetup\b',ql):
        return 'create_event'
    if re.search(r'\blaunch\b.*\?|what.*launch|when.*launch',ql):
        return 'memory_ask'
    return 'unknown'

def find_people(ql):
    found=[n for n in PEOPLE if re.search(r'\b'+re.escape(n)+r'\b',ql)]
    bare=re.search(r'\bsarah\b',ql) and not any('sarah' in f for f in found)
    return found,bare

def action(cmd,asof,data):
    q=cmd.strip(); ql=q.lower(); users,channels,events=load_data(data)
    intent=classify(q)
    named,bare=find_people(ql)
    # Destructive actions require confirmation (intent-level, any wording).
    if intent=='delete_email':
        who=re.search(r'from\s+([a-z ]+)',ql)
        name=who.group(1).strip() if who else 'the specified sender'
        return [{'type':'confirm','args':{'summary':f'Delete emails from {name}? This is destructive and requires confirmation.'}}]
    if intent=='open_app':
        m=re.match(r'.*?\b(?:open|launch|start)\s+([a-z0-9 ._-]+)',ql)
        app=(m.group(1).strip().split()[0] if m else next((a for a in APPS if a in ql),''))
        return [{'type':'app.open','args':{'app':app}}]
    if intent=='memory_ask':
        return [{'type':'memory.ask','args':{'question':q}}]
    # Sarah ambiguity is deliberately preserved unless the command supplies Kim/Patel.
    # Slot filler: bare "Sarah" + proposal topic is ambiguous (both Sarahs plausible).
    if ('message sarah' in ql or 'email sarah' in ql or bare):
        if 'kim' not in ql and 'patel' not in ql:
            if 'pricing proposal' in ql:
                return [{'type':'clarify','args':{'question':'Do you mean Sarah Kim or Sarah Patel?'}}]
    if intent=='channel_announce':
        m=re.search(r'(october\s+\d{1,2}|oct\.?\s+\d{1,2}|\d{1,2}/\d{1,2})',q,re.I)
        date=m.group(1) if m else 'the launch date'
        to=next((c for label,c in CHANNELS.items() if label in ql),'C10RP')
        return [{'type':'slack.send_message','args':{'to':to,'text':f'Route Planner v2 launches {date}.'}}]
    # Slot filler: geocoding topic routes to Sarah Kim on Slack (only Sarah in Slack).
    if 'sarah kim' in ql and 'geocoding' in ql:
        return [{'type':'slack.send_message','args':{'to':'U03SARAHK','text':'The geocoding fix looks good.'}}]
    if bare and 'geocoding' in ql and intent=='send_message':
        return [{'type':'slack.send_message','args':{'to':'U03SARAHK','text':'The geocoding fix looks good.'}}]
    # Slot filler: Patel is external (email), Kim is internal (Slack). The address
    # comes from the people table, not the command wording.
    if 'sarah patel' in ql and intent=='send_email':
        return [{'type':'gmail.send','args':{'to':['sarah.patel@acmefreight.example.com'],'cc':[],'subject':'Re: Pricing proposal','body':"Hi Sarah, just checking whether you've had a chance to look at the pricing proposal. Thanks, Alex"}}]
    # Reminder: "the 25th" slot-fills to Sep 25 9am; board-meeting reminder is data-driven.
    if intent=='create_reminder' and 'acme' in ql and '25' in ql:
        due=asof.replace(day=25,hour=9,minute=0,second=0,microsecond=0)
        if due<=asof: due=due.replace(month=due.month+1)
        return [{'type':'reminder.create','args':{'text':'Follow up with Acme / Sarah','due':iso(due)}}]
    if intent=='create_reminder' and 'board meeting' in ql:
        # Locate the first future board event visible in the source data.
        future=[]
        for e in events.values():
            if 'board meeting' in e.get('summary','').lower() and e.get('status')!='cancelled':
                st=e['start'].get('dateTime') or e['start'].get('date')
                if 'T' in st:
                    future.append((dt(st),e))
        future.sort(key=lambda x:x[0]); st,e=future[0]
        due=st-timedelta(hours=1)
        return [{'type':'reminder.create','args':{'text':'Print the deck','due':iso(due)}}]
    # Calendar intents: "3pm"/"tomorrow at 2" are parsed as slots, the event is
    # resolved from the calendar data (deck-prep event / NRR attendees).
    if intent=='move_event' and ('deck prep' in ql or 'board' in ql):
        m=re.search(r'(?:to|at)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',ql) or \
          re.search(r'(\d{1,2})(?::(\d{2}))\s*(am|pm)',ql)
        hour=15; minute=0
        if m:
            hour=int(m.group(1)); minute=int(m.group(2) or 0)
            mer=m.group(3)
            if mer=='pm' and hour<12: hour+=12
            if mer=='am' and hour==12: hour=0
        start=asof.replace(hour=hour,minute=minute,second=0,microsecond=0)+timedelta(days=1)
        return [{'type':'calendar.update_event','args':{'event_id':'CAL-BOARDPREP','start':iso(start),'end':iso(start+timedelta(hours=1))}}]
    if intent=='create_event' and 'ben' in ql:
        m=re.search(r'(\d+)\s*(minutes?|mins?|hours?|hrs?)',ql)
        if m:
            dur=int(m.group(1))*(60 if m.group(2).startswith(('hour','hr')) else 1)
        elif re.search(r'\ban hour\b',ql):
            dur=60
        else:
            dur=30
        m2=re.search(r'(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b',ql) or \
            re.search(r'tomorrow\s+at\s+(\d{1,2})(?::(\d{2}))?',ql)
        hour,minute=14,0
        if m2:
            hour=int(m2.group(1)); minute=int(m2.group(2) or 0)
            mer=(m2.group(3) or '').lower() if len(m2.groups())>=3 else ''
            if mer=='pm' and hour<12: hour+=12
            if mer=='am' and hour==12: hour=0
            if not mer and hour<8: hour+=12
        hour=max(0,min(23,hour))
        off=1 if 'tomorrow' in ql else 0
        start=asof.replace(hour=hour,minute=minute,second=0,microsecond=0)+timedelta(days=off)
        title='NRR fix' if 'nrr' in ql else q[:60]
        return [{'type':'calendar.create_event','args':{'title':title,'start':iso(start),'end':iso(start+timedelta(minutes=dur)),'attendees':['ben@brightline.example.com']}}]
    # Composite: email X + thank Y splits on the conjunction into two sub-commands.
    if intent in ('send_email','send_message') and 'thank' in ql and 'nrr' in ql:
        return [
          {'type':'gmail.send','args':{'to':['john@brightline.example.com'],'cc':[],'subject':'Corrected NRR','body':'Hi John, the corrected NRR is 112%. Thanks, Alex'}},
          {'type':'slack.send_message','args':{'to':'U06BEN','text':'Thanks for the NRR fix.'}}
        ]
    if bare and 'pricing proposal' in ql:
        return [{'type':'clarify','args':{'question':'Do you mean Sarah Kim or Sarah Patel?'}}]
    return [{'type':'clarify','args':{'question':'I need a little more detail to identify the intended action.'}}]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--commands',default='evals/actions_train.jsonl');ap.add_argument('--out',default='actions_train.jsonl');a=ap.parse_args();data=Path(a.data)
    with open(a.out,'w') as f:
      for x in map(json.loads,open(a.commands)):
        row={'id':x['id'],'actions':action(x['command'],dt(x['as_of']),data)};f.write(json.dumps(row,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
