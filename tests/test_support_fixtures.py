"""Self-tests for the shared fixtures in ``tests/support`` (fake model servers and fake MCP client)."""
import base64
import gzip
import http.client
import json
import select
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from support.fake_mcp_client import FakeMcpClient, McpError, parse_sse
from support.fake_model_servers import (FakeAnthropicServer, FakeChatServer, FakeResponsesServer, Failure, Gate, Step,
                                        Usage, text_step, tool_step)

PNG = b'\x89PNG\r\n\x1a\n' + bytes(range(32))
PNG_B64 = base64.b64encode(PNG).decode()


def post(server, path, body, headers=None, raw=None):
    conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
    data = raw if raw is not None else json.dumps(body).encode()
    conn.request('POST', path, body=data, headers={'Content-Type': 'application/json', **(headers or {})})
    resp = conn.getresponse()
    out = resp.status, resp.headers, resp.read()
    conn.close()
    return out


def events(raw):
    """SSE frames as (event, parsed data); data may be the string [DONE]."""
    out = []
    for frame in parse_sse(raw):
        if frame['data'] is None:
            continue
        data = frame['data'] if frame['data'] == '[DONE]' else json.loads(frame['data'])
        out.append((frame['event'], data))
    return out


class AnthropicTests(unittest.TestCase):
    def test_streaming_thinking_text_tool(self):
        step = Step(thinking='hmm', text='hello world, long text', stop_reason='tool_use', keepalives=1,
                    tool_calls=[('read_file', {'path': '/a/b.txt', 'n': 3})],
                    usage=Usage(7, 9, cache_read_input_tokens=100, cache_creation_input_tokens=5))
        with FakeAnthropicServer([step]) as server:
            status, headers, raw = post(server, '/v1/messages?beta=true', {'model': 'm1', 'stream': True,
                                                                          'messages': []})
        self.assertEqual(status, 200)
        self.assertIn('text/event-stream', headers['Content-Type'])
        evs = events(raw)
        kinds = [e for e, _ in evs]
        self.assertEqual(kinds[:2], ['message_start', 'ping'])
        self.assertEqual(kinds[-2:], ['message_delta', 'message_stop'])
        start = evs[0][1]['message']
        self.assertEqual((start['model'], start['usage']['cache_read_input_tokens'],
                          start['usage']['cache_creation_input_tokens']), ('m1', 100, 5))
        self.assertTrue(start['id'])
        starts = [d['content_block']['type'] for e, d in evs if e == 'content_block_start']
        self.assertEqual(starts, ['thinking', 'text', 'tool_use'])
        text = ''.join(d['delta']['text'] for e, d in evs if e == 'content_block_delta'
                       and d['delta']['type'] == 'text_delta')
        self.assertEqual(text, 'hello world, long text')
        sig = [d['delta']['signature'] for e, d in evs if e == 'content_block_delta'
               and d['delta']['type'] == 'signature_delta']
        self.assertEqual(sig, ['fake-signature-0001'])
        fragments = [d['delta']['partial_json'] for e, d in evs if e == 'content_block_delta'
                     and d['delta']['type'] == 'input_json_delta']
        self.assertGreater(len(fragments), 1)
        self.assertEqual(json.loads(''.join(fragments)), {'path': '/a/b.txt', 'n': 3})
        delta = evs[-2][1]
        self.assertEqual((delta['delta']['stop_reason'], delta['usage']['output_tokens']), ('tool_use', 9))
        for e, d in evs:
            if e.startswith('content_block'):
                self.assertIn('index', d)

    def test_non_streaming_and_routes(self):
        with FakeAnthropicServer([tool_step('x', {'a': 1}, id='toolu_1', text='t')], count_tokens_value=42) as server:
            status, _, raw = post(server, '/v1/messages', {'model': 'm', 'stream': False})
            body = json.loads(raw)
            self.assertEqual(status, 200)
            self.assertEqual([b['type'] for b in body['content']], ['text', 'tool_use'])
            self.assertEqual(body['content'][1], {'type': 'tool_use', 'id': 'toolu_1', 'name': 'x', 'input': {'a': 1}})
            self.assertEqual(body['stop_reason'], 'tool_use')
            self.assertEqual(json.loads(post(server, '/v1/messages/count_tokens', {})[2]), {'input_tokens': 42})
            conn = http.client.HTTPConnection('127.0.0.1', server.port)
            conn.request('GET', '/v1/models')
            self.assertEqual(conn.getresponse().status, 404)
            self.assertEqual([r.path for r in server.requests],
                             ['/v1/messages', '/v1/messages/count_tokens', '/v1/models'])
            self.assertEqual(len(server.generation_requests()), 1)

    def test_query_and_case_insensitive_headers(self):
        with FakeAnthropicServer() as server:
            post(server, '/v1/messages?beta=true', {'stream': False}, {'X-Api-Key': 'k'})
            req = server.requests[0]
            self.assertEqual(req.path, '/v1/messages?beta=true')
            self.assertEqual((req.route, req.query), ('/v1/messages', {'beta': ['true']}))
            self.assertEqual((req.headers['x-api-key'], req.header('X-API-KEY')), ('k', 'k'))

    def test_gzip_body_decoded(self):
        payload = {'model': 'm', 'stream': False, 'messages': [{'role': 'user', 'content': 'hi'}]}
        with FakeAnthropicServer() as server:
            post(server, '/v1/messages', None, {'Content-Encoding': 'gzip'}, raw=gzip.compress(
                json.dumps(payload).encode()))
            req = server.requests[0]
            self.assertEqual(req.json, payload)
            self.assertEqual(json.loads(req.body), payload)
            self.assertNotEqual(req.raw_body, req.body)
            self.assertEqual(server.last_json(), payload)

    def test_scripted_errors(self):
        cases = [(400, 'invalid_request_error'), (401, 'authentication_error'), (403, 'permission_error'),
                 (404, 'not_found_error'), (413, 'request_too_large'), (429, 'rate_limit_error'),
                 (500, 'api_error'), (529, 'overloaded_error')]
        script = [Step(failure=Failure(status=s, message=f'm{s}', retry_after=7)) for s, _ in cases]
        with FakeAnthropicServer(script) as server:
            for status, kind in cases:
                code, headers, raw = post(server, '/v1/messages', {'stream': True})
                self.assertEqual(code, status)
                self.assertEqual(json.loads(raw), {'type': 'error', 'error': {'type': kind, 'message': f'm{status}'}})
                if status == 429:
                    self.assertEqual(headers['retry-after'], '7')

    def test_stream_failures(self):
        script = [Step(text='abcdefghijklmnopqrstuvwxyz0123456789', failure=Failure('stream_error')),
                  Step(text='partial', failure=Failure('truncated')),
                  Step(tool_calls=[('t', {'k': 'v'})], failure=Failure('malformed_tool_json'))]
        with FakeAnthropicServer(script) as server:
            evs = events(post(server, '/v1/messages', {'stream': True})[2])
            self.assertEqual(evs[-1][0], 'error')
            self.assertEqual(evs[-1][1]['error']['type'], 'overloaded_error')
            self.assertNotIn('message_stop', [e for e, _ in evs])
            evs = events(post(server, '/v1/messages', {'stream': True})[2])
            self.assertNotIn('message_stop', [e for e, _ in evs])
            self.assertNotIn('message_delta', [e for e, _ in evs])
            evs = events(post(server, '/v1/messages', {'stream': True})[2])
            args = ''.join(d['delta']['partial_json'] for e, d in evs if e == 'content_block_delta')
            with self.assertRaises(ValueError):
                json.loads(args)
            self.assertEqual(evs[-1][0], 'message_stop')

    def test_gate_blocks_until_released(self):
        gate = Gate()
        with FakeAnthropicServer([Step(text='late', gate=gate)]) as server:
            result = {}
            thread = threading.Thread(target=lambda: result.update(r=post(server, '/v1/messages',
                                                                         {'stream': False})))
            thread.start()
            self.assertTrue(gate.wait_entered())
            self.assertEqual(len(server.wait_for_requests(1)), 1)
            time.sleep(0.05)
            self.assertNotIn('r', result)
            gate.release()
            thread.join(5)
            self.assertEqual(json.loads(result['r'][2])['content'][0]['text'], 'late')

    def test_exhausted_script_default_and_callable(self):
        def react(req):
            return text_step('saw ' + req.json['messages'][0]['content'])

        with FakeAnthropicServer([react]) as server:
            first = json.loads(post(server, '/v1/messages', {'messages': [{'content': 'x'}]})[2])
            self.assertEqual(first['content'][0]['text'], 'saw x')
            for _ in range(2):
                again = json.loads(post(server, '/v1/messages', {'stream': False})[2])
                self.assertEqual(again['content'][0]['text'], 'done')
            self.assertEqual(len(server.generation_requests()), 3)
            with self.assertRaises(TimeoutError):
                server.wait_for_requests(4, timeout=0.05)

    def test_images_and_tool_results(self):
        body = {'messages': [
            {'role': 'user', 'content': 'go'},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'toolu_9', 'name': 'obs', 'input': {}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_9', 'content': [
                {'type': 'text', 'text': 'frame'},
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': PNG_B64}}]}]}]}
        with FakeAnthropicServer() as server:
            post(server, '/v1/messages', body)
            image, = server.images()
            self.assertEqual((image.media_type, image.data), ('image/png', PNG))
            result, = server.find_tool_results()
            self.assertEqual((result.call_id, result.text, result.is_error), ('toolu_9', 'frame', False))
            self.assertEqual(result.images[0].sha256, image.sha256)


class ChatTests(unittest.TestCase):
    def test_streaming_with_usage(self):
        step = Step(text='hello there, friend', tool_calls=[('f', {'q': 'x' * 30}, 'call_1')],
                    usage=Usage(5, 6, cache_read_input_tokens=3, reasoning_tokens=2))
        with FakeChatServer([step]) as server:
            _, _, raw = post(server, '/chat/completions', {'model': 'm', 'stream': True,
                                                          'stream_options': {'include_usage': True}})
        evs = [d for _, d in events(raw)]
        self.assertEqual(evs[-1], '[DONE]')
        usage = evs[-2]
        self.assertEqual(usage['choices'], [])
        self.assertEqual(usage['usage']['prompt_tokens_details']['cached_tokens'], 3)
        self.assertEqual((usage['usage']['prompt_tokens'], usage['usage']['total_tokens']), (8, 14))
        self.assertEqual(usage['usage']['completion_tokens_details']['reasoning_tokens'], 2)
        chunks = evs[:-2]
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(chunks[0]['model'], 'm')
        text = ''.join(c['choices'][0]['delta'].get('content') or '' for c in chunks)
        self.assertEqual(text, 'hello there, friend')
        calls = [tc for c in chunks for tc in c['choices'][0]['delta'].get('tool_calls', [])]
        self.assertEqual((calls[0]['id'], calls[0]['type'], calls[0]['function']['name']), ('call_1', 'function', 'f'))
        self.assertEqual(json.loads(''.join(c['function']['arguments'] for c in calls)), {'q': 'x' * 30})

    def test_no_usage_chunk_without_option_and_non_streaming(self):
        with FakeChatServer([text_step('a'), tool_step('g', {'z': 1}, id='c9')]) as server:
            evs = [d for _, d in events(post(server, '/v1/chat/completions', {'stream': True})[2])]
            self.assertEqual(evs[-1], '[DONE]')
            self.assertTrue(all(c['choices'] for c in evs[:-1]))
            body = json.loads(post(server, '/v1/chat/completions', {})[2])
            message = body['choices'][0]['message']
            self.assertEqual(body['choices'][0]['finish_reason'], 'tool_calls')
            self.assertEqual(message['tool_calls'][0], {'id': 'c9', 'type': 'function',
                                                        'function': {'name': 'g', 'arguments': '{"z": 1}'}})

    def test_errors_and_truncation(self):
        script = [Step(failure=Failure(status=429, message='slow', retry_after=2)),
                  Step(text='abcdefghijklmnopqrstuvwxyz', failure=Failure('stream_error', message='boom')),
                  Step(text='abc', failure=Failure('truncated'))]
        with FakeChatServer(script) as server:
            code, headers, raw = post(server, '/v1/chat/completions', {'stream': True})
            self.assertEqual((code, headers['retry-after']), (429, '2'))
            self.assertEqual(json.loads(raw)['error']['type'], 'rate_limit_error')
            evs = [d for _, d in events(post(server, '/v1/chat/completions', {'stream': True})[2])]
            self.assertEqual(evs[-1]['error']['message'], 'boom')
            evs = [d for _, d in events(post(server, '/v1/chat/completions', {'stream': True})[2])]
            self.assertNotIn('[DONE]', evs)

    def test_tool_message_images(self):
        body = {'messages': [{'role': 'user', 'content': 'x'},
                             {'role': 'tool', 'tool_call_id': 'c1', 'content': [
                                 {'type': 'text', 'text': 'shot'},
                                 {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + PNG_B64}}]},
                             {'role': 'tool', 'tool_call_id': 'c2', 'content': 'plain'}]}
        with FakeChatServer() as server:
            post(server, '/v1/chat/completions', body)
            first, second = server.find_tool_results()
            self.assertEqual((first.call_id, first.text, second.text), ('c1', 'shot', 'plain'))
            self.assertEqual(server.images()[0].data, PNG)


class ResponsesTests(unittest.TestCase):
    def test_streaming(self):
        step = Step(thinking='t', text='some answer text', tool_calls=[('shell', {'cmd': 'ls -la /tmp'}, 'call_7')],
                    usage=Usage(4, 6, cache_read_input_tokens=2, reasoning_tokens=1))
        with FakeResponsesServer([step]) as server:
            _, _, raw = post(server, '/v1/responses', {'model': 'm', 'stream': True})
        evs = events(raw)
        kinds = [e for e, _ in evs]
        self.assertEqual(kinds[0], 'response.created')
        self.assertEqual(kinds[-1], 'response.completed')
        self.assertEqual([d['type'] for _, d in evs], kinds)
        self.assertLess(kinds.index('response.output_item.added'), kinds.index('response.output_text.delta'))
        self.assertLess(kinds.index('response.output_text.delta'), kinds.index('response.function_call_arguments.delta'))
        text = ''.join(d['delta'] for e, d in evs if e == 'response.output_text.delta')
        self.assertEqual(text, 'some answer text')
        args = ''.join(d['delta'] for e, d in evs if e == 'response.function_call_arguments.delta')
        self.assertEqual(json.loads(args), {'cmd': 'ls -la /tmp'})
        done, = [d for e, d in evs if e == 'response.function_call_arguments.done']
        self.assertEqual(json.loads(done['arguments']), {'cmd': 'ls -la /tmp'})
        response = evs[-1][1]['response']
        self.assertEqual((response['status'], response['model']), ('completed', 'm'))
        self.assertTrue(response['id'])
        self.assertEqual(response['usage']['input_tokens_details']['cached_tokens'], 2)
        self.assertEqual(response['usage']['output_tokens_details']['reasoning_tokens'], 1)
        call, = [i for i in response['output'] if i['type'] == 'function_call']
        self.assertEqual((call['call_id'], call['name']), ('call_7', 'shell'))

    def test_non_streaming_terminals_and_compact(self):
        script = [text_step('hi'), Step(text='cut', stop_reason='max_tokens'),
                  Step(failure=Failure('failed', message='nope')), Step(text='x', failure=Failure('incomplete'))]
        with FakeResponsesServer(script) as server:
            self.assertEqual(json.loads(post(server, '/responses', {})[2])['status'], 'completed')
            self.assertEqual(json.loads(post(server, '/responses', {})[2])['status'], 'incomplete')
            failed = events(post(server, '/responses', {'stream': True})[2])
            self.assertEqual(failed[-1][0], 'response.failed')
            self.assertEqual(failed[-1][1]['response']['output'], [])
            self.assertEqual(events(post(server, '/v1/responses', {'stream': True})[2])[-1][0],
                             'response.incomplete')
            compact = json.loads(post(server, '/responses/compact', {})[2])
            self.assertEqual(compact['object'], 'response.compaction')
            self.assertEqual(len(server.generation_requests()), 4)
            self.assertEqual(len(server.requests), 5)

    def test_errors_and_stream_error(self):
        script = [Step(failure=Failure(status=529, message='busy')), Step(text='abcdefghijklmnop' * 3,
                                                                          failure=Failure('stream_error'))]
        with FakeResponsesServer(script) as server:
            code, _, raw = post(server, '/responses', {'stream': True})
            self.assertEqual(code, 529)
            self.assertEqual(json.loads(raw)['error']['message'], 'busy')
            evs = events(post(server, '/responses', {'stream': True})[2])
            self.assertEqual(evs[-1][0], 'error')
            self.assertNotIn('response.completed', [e for e, _ in evs])

    def test_function_call_output_and_images(self):
        body = {'input': [{'type': 'function_call_output', 'call_id': 'call_1', 'output': 'text out'},
                          {'type': 'function_call_output', 'call_id': 'call_2', 'output': [
                              {'type': 'input_text', 'text': 'see'},
                              {'type': 'input_image', 'image_url': 'data:image/png;base64,' + PNG_B64}]},
                          {'type': 'message', 'role': 'user', 'content': [
                              {'type': 'input_image', 'image_url': 'https://example.invalid/x.png'}]}]}
        with FakeResponsesServer() as server:
            post(server, '/responses', body)
            one, two = server.find_tool_results()
            self.assertEqual((one.call_id, one.text, two.text), ('call_1', 'text out', 'see'))
            self.assertEqual([i.data for i in server.images()], [PNG])
            self.assertEqual(two.images[0].media_type, 'image/png')


# ---------------------------------------------------------------------------------------------------------------------
class McpStub:
    """Tiny MCP server: initialize / tools/list / tools/call (``echo`` JSON or SSE with progress, ``block`` waits)."""

    def __init__(self, token='tok'):
        self.token, self.received = token, []
        self.in_call, self.release, self.disconnected = threading.Event(), threading.Event(), threading.Event()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(405)
                self.send_header('Content-Length', '0')
                self.end_headers()

            do_DELETE = do_GET

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                stub.received.append((dict(self.headers), body))
                if self.headers.get('Authorization') != f'Bearer {stub.token}':
                    return self.reply(401, {'error': 'unauthorized'})
                method, rid = body.get('method'), body.get('id')
                if rid is None:
                    self.send_response(202)
                    self.send_header('Content-Length', '0')
                    return self.end_headers()
                if method == 'initialize':
                    result = {'protocolVersion': body['params']['protocolVersion'],
                              'capabilities': {'tools': {'listChanged': False}},
                              'serverInfo': {'name': 'stub', 'version': '1'}}
                elif method == 'ping':
                    result = {}
                elif method == 'tools/list':
                    result = {'tools': [{'name': 'echo', 'description': 'd', 'inputSchema': {'type': 'object'}}]}
                elif method == 'tools/call':
                    return self.call(rid, body['params'])
                else:
                    return self.reply(200, {'jsonrpc': '2.0', 'id': rid,
                                            'error': {'code': -32601, 'message': 'no such method'}})
                self.reply(200, {'jsonrpc': '2.0', 'id': rid, 'result': result})

            def reply(self, status, message):
                data = json.dumps(message).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def call(self, rid, params):
                result = {'content': [{'type': 'text', 'text': json.dumps(params['arguments'])}], 'isError': False}
                final = {'jsonrpc': '2.0', 'id': rid, 'result': result}
                if 'text/event-stream' not in self.headers.get('Accept', ''):
                    return self.reply(200, final)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                token = (params.get('_meta') or {}).get('progressToken')
                if token is not None:
                    note = {'jsonrpc': '2.0', 'method': 'notifications/progress',
                            'params': {'progressToken': token, 'progress': 1}}
                    self.wfile.write(b': keepalive\n\n' + f'event: message\ndata: {json.dumps(note)}\n\n'.encode())
                self.wfile.flush()
                if params['name'] == 'block':
                    stub.in_call.set()
                    while not stub.release.is_set():
                        readable, _, _ = select.select([self.connection], [], [], 0.01)
                        try:
                            gone = bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)
                        except OSError:  # RST: the client closed with unread data
                            gone = True
                        if gone:
                            stub.disconnected.set()
                            return
                self.wfile.write(f'event: message\ndata: {json.dumps(final)}\n\n'.encode())
                self.close_connection = True

        class Httpd(ThreadingHTTPServer):
            daemon_threads = True

        self.httpd = Httpd(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}/mcp'

    def close(self):
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


class McpClientTests(unittest.TestCase):
    def setUp(self):
        self.stub = McpStub()
        self.addCleanup(self.stub.close)
        self.client = FakeMcpClient(self.stub.url, token='tok')

    def test_handshake_ping_list(self):
        init = self.client.handshake()
        self.assertEqual(init['protocolVersion'], '2025-06-18')
        first = self.stub.received[0][1]
        self.assertEqual(first['params']['clientInfo'], {'name': 'fake-mcp-client', 'version': '0.0.0'})
        self.assertEqual(self.stub.received[1][1], {'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        self.assertEqual(self.client.last_response.status, 202)
        self.assertEqual(self.client.ping(), {})
        self.assertEqual(self.client.list_tools()[0]['name'], 'echo')
        self.assertEqual(self.stub.received[0][0]['Authorization'], 'Bearer tok')
        self.assertEqual(self.client.http('GET').status, 405)
        with self.assertRaises(McpError) as ctx:
            self.client._call('nope')
        self.assertEqual(ctx.exception.error['code'], -32601)

    def test_bad_token(self):
        res = FakeMcpClient(self.stub.url, token='wrong').call_tool('echo', {})
        self.assertEqual((res.status, res.result, res.message), (401, None, None))

    def test_call_json_and_sse(self):
        res = self.client.call_tool('echo', {'a': 1}, accept_sse=False)
        self.assertFalse(res.is_sse)
        self.assertEqual(json.loads(res.result['content'][0]['text']), {'a': 1})
        self.assertEqual(res.progress, [])
        self.assertEqual(self.stub.received[-1][0]['Accept'], 'application/json')
        res = self.client.call_tool('echo', {'b': 2}, progress_token='p1', meta={'claudecode/toolUseId': 'toolu_1'})
        self.assertTrue(res.is_sse)
        self.assertEqual(res.progress, [{'progressToken': 'p1', 'progress': 1}])
        self.assertEqual(json.loads(res.result['content'][0]['text']), {'b': 2})
        self.assertEqual(res.response.frames[0]['comments'], ['keepalive'])
        self.assertEqual(self.stub.received[-1][1]['params']['_meta'],
                         {'claudecode/toolUseId': 'toolu_1', 'progressToken': 'p1'})
        self.assertIn(b'notifications/progress', res.response.raw)
        self.assertEqual(self.client.exchanges[-1].request_headers['Accept'], 'application/json, text/event-stream')

    def test_cancel_while_running(self):
        pending = self.client.start_call('block', {'k': 1}, progress_token=5)
        self.assertTrue(self.stub.in_call.wait(5))
        ack = self.client.cancel(pending.request_id, 'user')
        self.assertEqual(ack.status, 202)
        self.assertEqual(self.stub.received[-1][1], {'jsonrpc': '2.0', 'method': 'notifications/cancelled',
                                                     'params': {'requestId': pending.request_id, 'reason': 'user'}})
        self.assertFalse(pending.done())
        self.stub.release.set()
        res = pending.wait(5)
        self.assertEqual(res.request_id, pending.request_id)
        self.assertFalse(res.result['isError'])

    def test_disconnect_mid_call(self):
        request_id = self.client.call_tool_and_disconnect('block', {}, progress_token='p')
        self.assertTrue(self.stub.in_call.wait(5))
        self.assertTrue(self.stub.disconnected.wait(5))
        self.assertEqual(self.stub.received[-1][1]['id'], request_id)


if __name__ == '__main__':
    unittest.main()
