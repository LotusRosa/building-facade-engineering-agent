"""Workflow entry changes must preserve the existing training specifications."""
import unittest
import test_workflow as fixtures
from facade_next.contracts import methods


class CoreEntryTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.WorkflowTests('test_hold_and_protected_evaluation')
        self.f.setUp()
        self.addCleanup(self.f.tearDown)

    def initial(self,tools):
        return self.f.call('create_round', project=self.f.p,
            train=[self.f.dataset('train')], val=[self.f.dataset('val')],
            evaluation=[self.f.dataset('safety')], backbone='resnet50', tools=tools, seed=7)['id']

    def test_public_nine_entries_core_preserves_standard_training_spec(self):
        entries=self.f.s.snapshot()['tools']
        self.assertEqual(entries[0],'core')
        self.assertEqual(len(entries),9)
        try:
            rid=self.initial(['core'])
        except ValueError as error:
            self.fail('Core initial request must be accepted: '+str(error))
        rnd=next(r for r in self.f.s.snapshot()['rounds'] if r['id']==rid)
        self.assertEqual(rnd['entry_tools'],['core'])
        self.assertEqual(rnd['tools'],['standard'])
        self.assertEqual(self.f.s.snapshot()['jobs'],[])
        self.f.call('start_round',round=rid)
        self.assertEqual(self.f.s.snapshot()['jobs'][0]['spec']['method'],methods()['standard'])

    def test_initial_empty_selection_defaults_to_core(self):
        try:
            rid=self.initial([])
        except ValueError as error:
            self.fail('Initial Core is mandatory/default, not an empty optimization selection: '+str(error))
        rnd=next(r for r in self.f.s.snapshot()['rounds'] if r['id']==rid)
        self.assertEqual(rnd['entry_tools'],['core'])
        self.assertEqual(rnd['tools'],['standard'])

    def test_legacy_initial_standard_remains_compatible_and_stage_named_core(self):
        rid=self.initial(['standard'])
        rnd=next(r for r in self.f.s.snapshot()['rounds'] if r['id']==rid)
        self.assertEqual(rnd.get('entry_tools'),['core'])
        # Older records need not be rewritten to render their initial entry.
        with self.f.s.store.transaction() as db:
            old=self.f.s.store.get(db,'round',rid);old.pop('entry_tools',None)
            self.f.s.store.put(db,'round',old)
        from facade_next.names import method_name
        self.assertIn('Core',method_name('standard','en',kind='initial'))
        self.assertIn('Standard',method_name('standard','en',kind='update'))
        self.assertEqual(next(r for r in self.f.s.snapshot()['rounds'] if r['id']==rid)['tools'],['standard'])

    def test_maintenance_core_is_rejected_and_standard_is_available(self):
        rid=self.f.round();jid=self.f.fake_complete(rid)
        self.f.call('decide',round=rid,job=jid,reason='initial test adoption')
        payload=dict(project=self.f.p, train=[self.f.dataset('new')],val=[],
                     evaluation=[self.f.dataset('gate')], backbone='resnet50', seed=7)
        with self.assertRaisesRegex(ValueError,'Core.*initial|initial.*Core'):
            self.f.call('create_round',**payload,tools=['core'])
        with self.assertRaisesRegex(ValueError,'maintenance'):
            self.f.call('create_round',**payload,tools=[])
        rid=self.f.call('create_round',**payload,tools=['standard','sam'])['id']
        rnd=next(r for r in self.f.s.snapshot()['rounds'] if r['id']==rid)
        self.assertEqual(rnd['tools'],['standard','sam'])

    def test_core_initial_is_not_an_additional_optimization_combination(self):
        with self.assertRaisesRegex(ValueError,'Core|initial'):
            self.initial(['core','standard'])

    def test_core_chat_alias_can_select_and_adopt_only_the_initial_candidate(self):
        from facade_next.llm_chat import validate
        try:
            value=validate(dict(intent='configure',arguments={'tools':['core']}),'use Core')
        except ValueError as error:
            self.fail('LLM Core selection rejected: '+str(error))
        self.assertEqual(value[1]['tools'],['core'])
        self.f.call('chat',project=self.f.p,text='use Core',language='en')
        self.assertEqual(self.f.s.snapshot()['conversations'][0]['context']['options']['tools'],['core'])
        rid=self.f.round();jid=self.f.fake_complete(rid)
        from facade_next.conversation import propose_decision
        with self.f.s.store.transaction() as db:
            _,card=propose_decision(self.f.s,db,dict(project=self.f.p,context={}),
                                   dict(choice='adopt',job='core',reason='test initial baseline'))
        self.assertEqual(card['kind'],'confirm_adopt')
        self.assertIsNone(self.f.s.snapshot()['projects'][0]['active'])

    def test_legacy_core_safety_data_name_is_not_a_method_selection(self):
        from facade_next.names import mentioned_methods
        self.assertEqual(mentioned_methods('Core Safety acceptance data'),[])

    def test_adoption_changes_inherited_core_to_standard_without_overwriting_maintenance_drafts(self):
        from facade_next.conversation import sync_closed_round,configure
        rid=self.initial(['core']);jid=self.f.fake_complete(rid)
        self.f.call('decide',round=rid,job=jid,reason='initial baseline')
        c=dict(project=self.f.p,context={'options':{'tools':['core']}})
        with self.f.s.store.transaction() as db:
            sync_closed_round(self.f.s,db,c)
            self.assertEqual(c['context']['options']['tools'],['standard'])
            c['context']['options']['tools']=['sam']
            sync_closed_round(self.f.s,db,c)
            self.assertEqual(c['context']['options']['tools'],['sam'])
        # Also recover an older persisted conversation that has already synced.
        c['context']['options']['tools']=['core']
        new=self.f.dataset('maintenance');gate=self.f.dataset('gate')
        with self.f.s.store.transaction() as db:
            configure(self.f.s,db,c,dict(train=[new],evaluation=[gate],val=[]))
        self.assertEqual(self.f.s.snapshot()['rounds'][-1]['tools'],['standard'])


if __name__=='__main__':unittest.main()
