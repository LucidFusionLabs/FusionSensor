/*
 * $Id$
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LFL_FS_FS_H__
#define LFL_FS_FS_H__
namespace LFL {

struct AcousticEventHeader { long long timestamp; int m, n; };

struct SpeechDecodeClient : public FeatureSink {
  FixedAllocator<65536> alloc;
  SocketServices *net;
  Connection *conn=0;
  string host, path, flags;
  long long wrote=0, lastTimestamp=0;
  int flagstate=0;
  Callback resetCB;

  SpeechDecodeClient(SocketServices *n, const Callback &rcb=Callback()) : net(n), resetCB(rcb) {}
  ~SpeechDecodeClient() { Reset(); }

  void Reset() { if (conn) conn->SetError(); conn=0; host.clear(); path.clear(); flags.clear(); flagstate=0; }

  int Connect(const char *url) {
    Reset();
    conn = HTTPClient::PersistentConnection(net, url, &host, &path, bind(&SpeechDecodeClient::HTTPClientResponseCB, this, _1, _2, _3, _4, _5));
    return 0;
  }

  bool Connected() { return conn && conn->state == Connection::Connected; }

  int Write(const Matrix *feat, long long timestamp, bool flush=0, FeatureSink::ResponseCB cb=0) {
    if (!Connected()) return 0;

    if (flagstate == 0) {
      flagstate = 1;
      if (HTTPClient::WriteRequest(conn, HTTPServer::Method::GET, host.c_str(), "flags", "application/octet-stream", 0, 0, true) < 0) conn->SetError();
      return 0;
    }
    if (flagstate == 1) return 0;

    alloc.Reset();
    int len = sizeof(AcousticEventHeader) + sizeof(float) * feat->M * feat->N;
    char *buf = static_cast<char*>(alloc.Malloc(len));

    AcousticEventHeader *AEH = reinterpret_cast<AcousticEventHeader*>(buf);
    AEH->timestamp = timestamp;
    AEH->m = feat->M;
    AEH->n = feat->N;

    float *v = reinterpret_cast<float*>(AEH+1);
    MatrixIter(feat) v[i*feat->N + j] = feat->row(i)[j];

    string url = path;
    if (flush) url = StrCat(path, "?flush=1");

    int lbw = conn->wb.size();
    if (HTTPClient::WriteRequest(conn, HTTPServer::Method::POST, host.c_str(), url.c_str(), "application/octet-stream", buf, len, true) < 0) {
      conn->SetError();
      return 0;
    }
    wrote += conn->wb.size() - lbw;
    lastTimestamp = timestamp;
    responseCB = cb;
    inputlen = feat->M;
    return len;
  }

  void Flush() {
    if (!Connected()) return;

    alloc.Reset();
    int len = sizeof(AcousticEventHeader);
    char *buf = static_cast<char*>(alloc.Malloc(len));

    AcousticEventHeader *AEH = reinterpret_cast<AcousticEventHeader*>(buf);
    memset(AEH, 0, sizeof(AcousticEventHeader));

    int lbw = conn->wb.size();
    if (HTTPClient::WriteRequest(conn, HTTPServer::Method::POST, host.c_str(), path.c_str(), "application/octet-stream", buf, len, true) < 0) {
      conn->SetError();
      return;
    }
    wrote += conn->wb.size() - lbw;
  }

  void Read(Connection *c, const char *content, int content_length) {
    if (!content) {
      if (c == conn) {
        INFO("SpeechDecodeClient: lost connection: ", host);
      }
      return;
    }

    if (flagstate == 1) {
      flagstate = 2;
      flags = string(content, content_length);
      AcousticModel::LoadFlags(flags.c_str());
      if (resetCB) resetCB();
      return;
    }

    const AcousticEventHeader *AEHin = reinterpret_cast<const AcousticEventHeader*>(content);
    int AERlen = sizeof(AcousticEventHeader) + AEHin->m * AEHin->n * sizeof(float);
    if (content_length < AERlen) { ERROR("corrupt response ", content_length, " < ", AERlen); return; }

    alloc.Reset();
    AcousticEventHeader *AEH = reinterpret_cast<AcousticEventHeader*>(alloc.Malloc(AERlen));
    memcpy(AEH, content, AERlen);

    if (!AEH->m || AEH->n != 2) return;
    float *timestamp = reinterpret_cast<float*>(AEH+1);
    const char *transcript = content + AERlen;
    long long start = AEH->timestamp + timestamp[0];

    /* if rewrite clear space */
    while (decode.size() && decode.back().end >= start) decode.pop_back();

    string tsa;
    StringWordIter ts(transcript);
    for (int i=0; i<AEH->m; i++) {
      string word = ts.NextString();
      if (ts.Done()) continue;

      long long ts = AEH->timestamp + timestamp[i*2], ts2 = AEH->timestamp + timestamp[i*2+1];
      tsa += word + StringPrintf("-%lld(%lld, %f, %f) ", ts, AEH->timestamp, timestamp[i*2], timestamp[i*2+1]);
      decode.push_back(DecodedWord(word.c_str(), ts, ts2));
    }
    int len = (FLAGS_sample_rate/FLAGS_feat_hop)*FLAGS_sample_secs;
    INFOf("-- %s (range = %lld - %lld = %d) ds = %d", tsa.c_str(), lastTimestamp-len, lastTimestamp, len, decode.size());

    if (responseCB) responseCB(decode, inputlen);
    responseCB = 0;
    inputlen = 0;
  }

  int HTTPClientResponseCB(Connection *c, const char *h, const string &ct, const char *cb, int cl) { Read(c, cb, cl); return 0; }
};

}; // namespace LFL
#endif // LFL_FS_FS_H__
