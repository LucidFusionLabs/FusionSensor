/*
 * $Id: fv.cpp 1336 2014-12-08 09:29:59Z justin $
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

#include "core/app/app.h"
#include "core/app/gui.h"
#include "core/app/network.h"
#include "core/app/camera.h"
#include "core/app/audio.h"
#include "core/ml/hmm.h"
#include "core/speech/speech.h"   
#include "core/speech/voice.h"
#include "fs.h"
#include "core/speech/aed.h"

#ifdef LFL_OPENCV
#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include "core/app/bindings/opencv.h"
#endif

namespace LFL {
DEFINE_float(clip, -30, "Clip audio under N decibels");
DEFINE_string(plot, "sg", "Audio GUI Plot [sg, zcr, pe]");
DEFINE_bool(camera_effects, true, "Render camera effects");
DEFINE_int(camera_orientation, 3, "Camera orientation");
DEFINE_string(speech_client, "auto", "Speech client send [manual, auto, flood]");

struct MyAppState {
  HTTPServer *httpd=0;
} *my_app;

struct LiveSpectogram {
  int feature_rate, feature_width;
  int samples_processed=0, samples_available=0, texture_slide=0;
  float vmax=35, scroll=0;
  RingSampler buf;
  RingSampler::RowMatHandle handle;
  unique_ptr<Matrix> transform;
  Asset *live, *snap;

  LiveSpectogram(Asset *L, Asset *S) :
    feature_rate(FLAGS_sample_rate/FLAGS_feat_hop), feature_width(FLAGS_feat_window/2),
  buf(feature_rate, feature_rate*FLAGS_sample_secs, feature_width*sizeof(double)), handle(&buf), live(L), snap(S) {
    live->tex.CreateBacked(feature_width, feature_rate*FLAGS_sample_secs);
    snap->tex.CreateBacked(feature_width, feature_rate*FLAGS_sample_secs);
  }

  int Behind() const { return samples_available - samples_processed; }

  void Resize(int width) {
    feature_width = width;
    buf.Resize(feature_rate, feature_rate*FLAGS_sample_secs, feature_width*sizeof(double));
    handle.Init();
    live->tex.Resize(feature_width, feature_rate*FLAGS_sample_secs);
    snap->tex.Resize(feature_width, feature_rate*FLAGS_sample_secs);
  }

  void XForm(const string &n) {
    if (n == "mel") {
      transform = unique_ptr<Matrix>(Features::FFT2Mel(FLAGS_feat_melbands, FLAGS_feat_minfreq, FLAGS_feat_maxfreq, FLAGS_feat_window, FLAGS_sample_rate)->Transpose(mDelA));
      Resize(FLAGS_feat_melbands);
    } else {
      transform.reset();
      Resize(FLAGS_feat_window/2);
    }
  }

  float Update(unsigned samples) {
    samples_available += samples;
    while(samples_available >= samples_processed + FLAGS_feat_window) {
      const double *last = handle.Read(-1)->row(0);
      RingSampler::Handle L(app->audio->IL.get(), app->audio->RL.next-Behind(), FLAGS_feat_window);

      Matrix *Frame, *FFT;
      Frame = FFT = Spectogram(&L, transform ? 0 : handle.Write(), FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window, vector<double>(), PowerDomain::abs);
      if (transform) Frame = Matrix::Mult(FFT, transform.get(), handle.Write());
      if (transform) delete FFT;

      int ind = texture_slide * live->tex.width * Pixel::Size(live->tex.pf);
      glSpectogram(app->focused->gd, Frame, live->tex.buf+ind, live->tex.pf, 1, feature_width, feature_width,
                   vmax, FLAGS_clip, 0, PowerDomain::abs);

      live->tex.Bind();
      live->tex.UpdateGL(Box(0, texture_slide, live->tex.width, 1));

      samples_processed += FLAGS_feat_hop;
      scroll = float(texture_slide + 1) / (feature_rate * FLAGS_sample_secs);
      texture_slide = (texture_slide + 1) % (feature_rate * FLAGS_sample_secs);
    }
    return -scroll;
  }

  void Draw(Box win, bool onoff, bool fullscreen, AcousticEventDetector *AED) {
    GraphicsContext gc(app->focused->gd);
    int orientation = fullscreen ? 3 : 5;
    StringWordIter plot_iter(FLAGS_plot);
    for (string ploti = plot_iter.NextString(); !plot_iter.Done(); ploti = plot_iter.NextString()) { 
      if (ploti == "sg") {
        if (onoff) live->tex.DrawCrimped(gc.gd, win, orientation, 0, -scroll);
        else       snap->tex.DrawCrimped(gc.gd, win, orientation, 0, 0);
      }
      else if (ploti == "wf") {
        static int decimate = 10;
        int delay = Behind(), samples = feature_rate*FLAGS_feat_hop*FLAGS_sample_secs;
        if (onoff) {
          RingSampler::Handle B(app->audio->IL.get(), app->audio->RL.next-delay, samples);
          Waveform::Decimated(win.Dimension(), &Color::white, &B, decimate).Draw(&gc, win); 
        }
        else {
          SoundAsset *snap_sound = app->soundasset("snap");
          RingSampler::Handle B(snap_sound->wav.get(), snap_sound->wav->ring.back, samples);
          Waveform::Decimated(win.Dimension(), &Color::white, &B, decimate).Draw(&gc, win); 
        }
      }
      else if (ploti == "pe" && onoff && AED) {
        RingSampler::Handle B(&AED->pe, AED->pe.ring.back);
        Waveform(win.Dimension(), &Color::white, &B).Draw(&gc, win);
      }
      else if (ploti == "zcr" && onoff && AED) {
        RingSampler::Handle B(&AED->zcr, AED->zcr.ring.back);
        Waveform(win.Dimension(), &Color::white, &B).Draw(&gc, win);
      }
      else if (ploti == "szcr" && onoff && AED) {
        RingSampler::DelayHandle B(&AED->szcr, AED->szcr.ring.back, AcousticEventDetector::szcr_shift);
        Waveform(win.Dimension(), &Color::white, &B).Draw(&gc, win);
      }
    }
  }

  void ProgressBar(Box win, float percent) {
    if (percent < 0 || percent > 1) return;

    Geometry *geom = new Geometry(GraphicsDevice::Lines, 2, NullPointer<v2>(), 0, 0, Color(1.0,1.0,1.0));
    v2 *vert = reinterpret_cast<v2*>(&geom->vert[0]);

    float xi = win.percentX(percent);
    vert[0] = v2(xi, win.y);
    vert[1] = v2(xi, win.y+win.h);

    auto gd = app->focused->gd;
    gd->DisableTexture();
    Scene::Select(gd, geom);
    Scene::Draw(gd, geom, 0);
    delete geom;
  }
};

struct LiveCamera {
  Asset *camera, *camfx;
  LiveCamera(Asset *C, Asset *F) : camera(C), camfx(F) {
    camera->tex.CreateBacked(256, 256, Pixel::RGB24);
    camfx ->tex.CreateBacked(256, 256, Pixel::GRAY8);
  }

  Asset *Input() {
    Asset *cam = 0;
    if (!cam) cam = app->asset_loader->movie_playing ? &app->asset_loader->movie_playing->video : 0;
    if (!cam) cam = FLAGS_enable_camera ? camera : 0;
    if (!cam) cam = app->asset("browser");
    return cam;
  }

  void Update() {
    Asset *cam = Input();
    if (camfx->tex.width != cam->tex.width || camfx->tex.height != cam->tex.height)
      camfx->tex.Resize(cam->tex.width, cam->tex.height);

    if (FLAGS_enable_camera && cam == camera) {
      /* update camera buffer */
      cam->tex.UpdateBuffer(app->camera->state.image, point(FLAGS_camera_image_width, FLAGS_camera_image_height),
                            app->camera->state.image_format, app->camera->state.image_linesize, Texture::Flag::Resample);

      /* flush camera buffer */
      cam->tex.Bind();
      cam->tex.UpdateGL();
    }

#ifdef LFL_OPENCV
    if (!FLAGS_camera_effects) return;

    /* copy greyscale image */
    IplImage camI, camfxI;
    TextureToIplImage(camfx->tex, &camfxI);
    TextureToIplImage(cam->tex, &camI);
    cvCvtColor(&camI, &camfxI, CV_RGB2GRAY);

    if (0) {
      /* apply gaussian filter */
      cvSmooth(&camfxI, &camfxI, CV_GAUSSIAN, 11, 11);
    }

    /* perform canny edge detection */
    cvCanny(&camfxI, &camfxI, 10, 100, 3);

    /* flush camfx buffer */
    camfx->tex.Bind();
    camfx->tex.UpdateGL();
#endif
  }

  void Draw(Box camw, Box fxw) {
    auto gd = app->focused->gd;
    Input()->tex.DrawCrimped(gd, camw, FLAGS_camera_orientation, 0, 0);
    camfx  ->tex.DrawCrimped(gd, fxw,  FLAGS_camera_orientation, 0, 0);
  }
};

struct MyWindowState {
  int myTab=1;
  bool decoding=0, myMonitor=1;
  unique_ptr<AcousticEventDetector> AED;
  PhoneticSegmentationGUI *segments=0;
  unique_ptr<LiveCamera> liveCam;
  unique_ptr<LiveSpectogram> liveSG;
  HTTPServer::StreamResource *stream = 0;
};

struct AudioGUI : public GUI {
  MyWindowState *ws;
  int last_audio_count=1;
  bool monitor_hover=0, decode=0, last_decode=0;
  FontRef norm_font, text_font;
  Widget::Button play_button, decode_button, record_button;
  unique_ptr<VoiceModel> voice;
  AcousticModel::Compiled *decodeModel = 0;
  string transcript, speech_client_last=FLAGS_speech_client;

  AudioGUI(Window *W, MyWindowState *S) : GUI(W), ws(S),
    norm_font(FontDesc(FLAGS_font, "", 12, Color::grey70, Color::black)),
    text_font(FontDesc(FLAGS_font, "", 8,  Color::grey80, Color::black)),
    play_button  (this, Singleton<DrawableNop>::Get(), "play",    MouseController::CB([=](){ if (!app->audio->Out.size()) W->shell->play(vector<string>(1,"snap")); })),
    decode_button(this, Singleton<DrawableNop>::Get(), "decode",  MouseController::CB([=](){ decode = true; })),
    record_button(this, Singleton<DrawableNop>::Get(), "monitor", MouseController::CB([=](){ ws->myMonitor = !ws->myMonitor; })) {
      play_button.v_align = decode_button.v_align = record_button.v_align = VAlign::Bottom;
      play_button.v_offset = decode_button.v_offset = -norm_font->Height();
  }

  void HandleNetDecodeResponse(FeatureSink::DecodedWords &decode, int responselen) {
    transcript.clear();
    for (int i=0, l=decode.size(); i<l; i++) transcript += (!i ? "" : "  ") + decode[i].text;
    INFO("transcript: ", transcript);
    if (ws->segments) root->DelGUIPointer(&ws->segments);
    ws->segments = root->AddGUI(make_unique<PhoneticSegmentationGUI>(root, decode, responselen, "snap"));
    ws->decoding = 0;
  }

  void Layout() {
    ResetGUI();
    box = root->Box();
    CHECK(norm_font.Load());
    CHECK(text_font.Load());
    Flow flow(&box, 0, &child_box);
    flow.layout.append_only = flow.cur_attr.blend = true;
    play_button  .LayoutBox(&flow, norm_font, root->Box(0,  -.85, .5, .15, .16, .0001).center(app->asset("but1")->tex.Dimension()));
    decode_button.LayoutBox(&flow, norm_font, root->Box(.5, -.85, .5, .15, .16, .0001).center(app->asset("but1")->tex.Dimension()));
    record_button.LayoutBox(&flow, norm_font, root->Box(0,  -.65,  1, .2,  .38, .0001).center(app->asset("onoff1")->tex.Dimension()));

    play_button.AddHoverBox(record_button.box, MouseController::CB([=](){ monitor_hover = !monitor_hover; }));
    play_button.AddHoverBox(record_button.box, MouseController::CB([=](){ if ( ws->myMonitor) MonitorCmd(vector<string>(1,"snap")); }));
    play_button.AddClickBox(record_button.box, MouseControllerCallback([=](){ if (!ws->myMonitor) SnapCmd(vector<string>(1,"snap")); }, true));
    // static Widget::Button fullscreenButton(&gui, 0, 0, sb, 0, setMyTab, (void*)4);
  }

  void Draw() {
    GUI::Draw();
    GraphicsContext gc(root->gd);
    last_audio_count = app->audio->Out.size();
    if (decode && last_decode) { if (!ws->myMonitor) DecodeCmd(vector<string>(1, "snap")); decode=0; }
    last_decode = decode;

    if (auto b = play_button  .GetDrawBox()) b->drawable = &app->asset(app->audio->Out.size() ? "but1" : "but0")->tex;
    if (auto b = decode_button.GetDrawBox()) b->drawable = &app->asset((decode || ws->decoding)  ? "but1" : "but0")->tex;
    if (auto b = record_button.GetDrawBox()) b->drawable = &app->asset(ws->myMonitor ? (monitor_hover ? "onoff1hover" : "onoff1") :
                                                                       (monitor_hover ? "onoff0hover" : "onoff0"))->tex;

    /* live spectogram */
    float yp=0.6, ys=0.4, xbdr=0.05, ybdr=0.07;
    Box sb = root->Box(0, yp, 1, ys, xbdr, ybdr);
    Box si = root->Box(0, yp, 1, ys, xbdr+0.04, ybdr+0.035);

    app->asset("sbg")->tex.Draw(&gc, sb);
    ws->liveSG->Draw(si, ws->myMonitor, false, ws->AED.get());
    app->asset("sgloss")->tex.Draw(&gc, sb);

    /* progress bar */
    if (app->audio->Out.size() && !ws->myMonitor) {
      float percent=1-float(app->audio->Out.size()+Audio::VisualDelay()*FLAGS_sample_rate*FLAGS_chans_out)/app->audio->outlast;
      ws->liveSG->ProgressBar(si, percent);
    }

    /* acoustic event frame */
    if (ws->myMonitor && ws->AED && (ws->AED->words.size() || SpeechClientFlood()))
      AcousticEventGUI::Draw(ws->AED.get(), si);

    /* segmentation */
    if (!ws->myMonitor && ws->segments) ws->segments->Frame(si, norm_font);

    /* transcript */
    if (transcript.size() && !ws->myMonitor) text_font->Draw(StrCat("transcript: ", transcript), point(root->width*.05, root->height*.05));
    else text_font->Draw(StringPrintf("press tick for console - FPS = %.2f - CR = %.2f", root->fps.FPS(), app->camera->fps.FPS()), point(root->width*.05, root->height*.05));

    /* f0 */
    if (0) {
      RingSampler::Handle f0in(app->audio->IL.get(), app->audio->RL.next-FLAGS_feat_window, FLAGS_feat_window);
      text_font->Draw(StringPrintf("hz %.0f", FundamentalFrequency(&f0in, FLAGS_feat_window, 0)), point(root->width*.85, root->height*.05));
    }

    /* SNR */
    if (ws->AED) text_font->Draw(StringPrintf("SNR = %.2f", ws->AED->SNR()), point(root->width*.75, root->height*.05));
  }

  void SpectogramTransformCmd(const vector<string> &args) { ws->liveSG->XForm(IndexOrDefault(args, 0).c_str()); }
  void MonitorCmd(const vector<string>&) { FLAGS_speech_client = speech_client_last; }
  void SpeechClientCmd(const vector<string> &args) {
    if (args.empty() || args[0].empty()) return;
    FLAGS_speech_client = speech_client_last = args[0];
  }

  void ServerCmd(const vector<string> &args) {
    if (args.empty() || args[0].empty()) { INFO("eg: server 192.168.2.188:4044/sink"); return; }
    dynamic_cast<SpeechDecodeClient*>(ws->AED->sink)->Connect(args[0].c_str());
  }

  void DrawCmd(const vector<string> &args) {
    Asset *a = app->asset(IndexOrDefault(args, 0));
    SoundAsset *sa = app->soundasset(IndexOrDefault(args, 0));
    if (!a || !sa) return;
    bool recalibrate=1;
    RingSampler::Handle H(sa->wav.get());
    glSpectogram(root->gd, &H, &a->tex, ws->liveSG->transform.get(), recalibrate?&ws->liveSG->vmax:0, FLAGS_clip);
  }

  void SnapCmd(const vector<string> &args) {
    Asset *a = app->asset(IndexOrDefault(args, 0));
    SoundAsset *sa = app->soundasset(IndexOrDefault(args, 0));
    if (!FLAGS_enable_audio || !a || !sa) return;

    FLAGS_speech_client = "manual";
    transcript.clear();
    if (ws->segments) root->DelGUIPointer(&ws->segments);

    app->audio->Snapshot(sa);
    DrawCmd(args);

    if (ws->AED && ws->AED->sink) {
      FeatureSink::DecodedWords decode;
      for (int i=0; i<ws->AED->sink->decode.size(); i++) {
        FeatureSink::DecodedWord *word = &ws->AED->sink->decode[i];
        double perc = ws->AED->Percent(word->beg);
        if (perc < 0) continue;
        if (perc > 1) break;
        int beg = perc * ws->AED->feature_rate * FLAGS_sample_secs;
        int end = beg + (word->end - word->beg);
        decode.push_back(FeatureSink::DecodedWord(word->text.c_str(), beg, end));
        transcript += (!i ? "" : "  ") + word->text;
      }
      if (decode.size()) {
        if (ws->segments) root->DelGUIPointer(&ws->segments);
        ws->segments = root->AddGUI(make_unique<PhoneticSegmentationGUI>(root, decode, ws->AED->feature_rate * FLAGS_sample_secs, "snap"));
      }
    }
  }

  void DecodeCmd(const vector<string> &args) {
    SoundAsset *sa = app->soundasset(IndexOrDefault(args, 0));
    if (!sa) return INFO("decode <assset>");
    if (ws->AED && ws->AED->sink && ws->AED->sink->Connected()) return NetDecodeCmd(args);
#ifndef LFL_MOBILE
    if (!decodeModel)
#endif
    { return ERROR("decode error: not connected to server: ", -1); }

    unique_ptr<Matrix> features(Features::FromAsset(sa, Features::Flag::Full));
    unique_ptr<Matrix> viterbi(Decoder::DecodeFeatures(decodeModel, features.get(), 1024));
    transcript = Decoder::Transcript(decodeModel, viterbi.get());
    if (ws->segments) root->DelGUIPointer(&ws->segments);
    ws->segments = root->AddGUI(make_unique<PhoneticSegmentationGUI>(root, decodeModel, viterbi.get(), "snap"));
    if (transcript.size()) INFO(transcript);
    else                   INFO("decode failed");
  }

  void NetDecodeCmd(const vector<string> &args) {
    SoundAsset *sa = app->soundasset(IndexOrDefault(args, 0));
    if (!sa) { INFO("decode <assset>"); return; }
    if (!ws->AED || !ws->AED->sink || !ws->AED->sink->Connected()) { INFO("not connected"); return; }

    if (ws->segments) root->DelGUIPointer(&ws->segments);
    Matrix *features = Features::FromAsset(sa, Features::Flag::Storable);
    ws->AED->sink->decode.clear();
    int posted = ws->AED->sink->Write(features, 0, true, bind(&AudioGUI::HandleNetDecodeResponse, this, _1, _2));
    delete features;
    INFO("posted decode len=", posted);
    ws->decoding = 1;
  }

  void SynthCmd(const vector<string> &args) {
    if (!voice) return;
    unique_ptr<RingSampler> synth(voice->Synth(Join(args, " ").c_str()));
    if (!synth) return ERROR("synth ret 0");
    INFO("synth ", synth->ring.size);
    RingSampler::Handle B(synth.get());
    app->audio->QueueMixBuf(&B);
  }

  void ResynthCmd(const vector<string> &args) {
    SoundAsset *sa = app->soundasset(args.size()?args[0]:"snap");
    if (sa) { Resynthesize(app->audio.get(), sa); }
  }
};

struct VideoGUI : public GUI {
  MyWindowState *ws;
  VideoGUI(Window *W, MyWindowState *S) : GUI(W), ws(S) { StartCameraCmd(vector<string>()); }

  void StartCameraCmd(const vector<string>&) {
    if (!FLAGS_enable_camera) {
      FLAGS_enable_camera = true;
      if (app->camera->Init()) { FLAGS_enable_camera = false; return INFO("camera init failed"); }
      INFO("camera started");
    }
    ws->liveCam = make_unique<LiveCamera>(app->asset("camera"), app->asset("camfx"));
  }

  void Draw() {
    if (ws->liveCam) {
      if (app->camera->state.have_sample || app->asset_loader->movie_playing) ws->liveCam->Update();

      float yp=0.5, ys=0.5, xbdr=0.05, ybdr=0.07;
      Box st = root->Box(0, .5, 1, ys, xbdr+0.04, ybdr+0.035, xbdr+0.04, .01);
      Box sb = root->Box(0, 0,  1, ys, xbdr+0.04, .01, xbdr+0.04, ybdr+0.035);
      ws->liveCam->Draw(st, sb);
    }

    static Font *text = app->fonts->Get(FLAGS_font, "", 9, Color::grey80, Color::black);
    text->Draw(StringPrintf("press tick for console - FPS = %.2f - CR = %.2f", root->fps.FPS(), app->camera->fps.FPS()), point(root->width*.05, root->height*.05));
   }
};

struct RoomGUI {
  MyWindowState *ws;
  Scene scene;
  RoomGUI(MyWindowState *S) : ws(S) {
    scene.Add(new Entity("axis",  app->asset("axis")));
    scene.Add(new Entity("grid",  app->asset("grid")));
    scene.Add(new Entity("room",  app->asset("room")));
    scene.Add(new Entity("arrow", app->asset("arrow"), v3(1,.24,1)));
  }

  int Frame(LFL::Window *W, unsigned clicks, int flag) {
    scene.cam.Look(W->gd);
    scene.Get("arrow")->YawRight(clicks);	
    scene.Draw(W->gd, &app->asset.vec);
    return 0;
  }
};

struct FullscreenGUI : public GUI {
  MyWindowState *ws;
  bool decode=0, lastdecode=0, close=0;
  int monitorcount=0;
  FontRef norm_font, text_font;
  Widget::Button play_button, close_button, decode_icon, fullscreen_button;

  FullscreenGUI(Window *W, MyWindowState *S) : GUI(W), ws(S),
    norm_font(FontDesc(FLAGS_font, "", 12, Color::grey70)),
    text_font(FontDesc(FLAGS_font, "", 12, Color::white, Color::clear, FontDesc::Outline)),
    play_button      (this, 0, "", MouseController::CB([=](){ ws->myMonitor=1; })),
    close_button     (this, 0, "", MouseController::CB([=](){ close=1; })),
    decode_icon      (this, 0, "", MouseController::CB()),
    fullscreen_button(this, 0, "", MouseController::CB([=](){ decode=1; })) {}

  void Layout() {
    box = root->Box();
    CHECK(norm_font.Load());
    CHECK(text_font.Load());
    Flow flow(&box, 0, &child_box);
    play_button      .LayoutBox(&flow, 0, root->Box(0,    -1, .09, .07));
    close_button     .LayoutBox(&flow, 0, root->Box(0,  -.07, .09, .07));
    fullscreen_button.LayoutBox(&flow, 0, root->Box());
    fullscreen_button.GetHitBox().run_only_if_first = true;
    // play_button.AddClickBox(play_button.box, MouseController::CB([=](){ if (!myMonitor) MyMonitor(vector<string>(1,"snap")); }));
  }

  void Draw() {
    GUI::Draw();
    GraphicsContext gc(root->gd);
    if (ws->myMonitor) monitorcount++;
    else monitorcount = 0;

    if (decode && lastdecode) {
      if (monitorcount > 70) {
        ws->myMonitor = false;
        // MySnap(vector<string>(1, "snap"));
        // MyDecode(vector<string>(1, "snap"));
      }
      decode = 0;
    }
    lastdecode = decode;

    ws->liveSG->Draw(root->Box(), ws->myMonitor, true, ws->AED.get());

    if (ws->myMonitor) {
      if (ws->AED && (ws->AED->words.size() || SpeechClientFlood())) AcousticEventGUI::Draw(ws->AED.get(), root->Box(), true);
    } else if (ws->segments) {
      ws->segments->Frame(root->Box(), norm_font, true);

      int total = ws->AED->feature_rate * FLAGS_sample_secs;
      for (auto it = ws->segments->segments.begin(); it != ws->segments->segments.end(); it++)
        text_font->Draw(it->name, point(box.centerX(), box.percentY(float(it->beg)/total)), 0, Font::DrawFlag::Orientation(3));
    }

    if (decode || ws->decoding)
      decode_icon.image->Draw(&gc, root->Box(.9, -.07, .09, .07));
  }
};

struct FVGUI : public GUI {
  FontRef font;
  Widget::Button tab1, tab2, tab3;
  AudioGUI *audio_gui=0;
  VideoGUI *video_gui=0;
  unique_ptr<RoomGUI> room_gui;
  FullscreenGUI *fullscreen_gui=0;
  MyWindowState ws;

  FVGUI(Window *W) : GUI(W), font(FontDesc(FLAGS_font, "", 12, Color::grey70, Color::black)),
  tab1(this, 0, "audio gui",  MouseController::CB(bind(&FVGUI::SetMyTab, this, 1))),
  tab2(this, 0, "video gui",  MouseController::CB(bind(&FVGUI::SetMyTab, this, 2))), 
  tab3(this, 0, "room model", MouseController::CB(bind(&FVGUI::SetMyTab, this, 3))) {
    Activate();
    tab1.outline_topleft     = tab2.outline_topleft     = tab3.outline_topleft     = &Color::grey80;
    tab1.outline_bottomright = tab2.outline_bottomright = tab3.outline_bottomright = &Color::grey40;
  }

  void SetMyTab(int a) { 
    ws.myTab = a; 
    audio_gui     ->active = ws.myTab == 1;
    fullscreen_gui->active = ws.myTab == 4;
  }

  void ReloadAED() { 
    ws.AED = make_unique<AcousticEventDetector>(FLAGS_sample_rate/FLAGS_feat_hop,
                                                new SpeechDecodeClient(bind(&FVGUI::ReloadAED, this)));
    ws.AED->AlwaysComputeFeatures();
  }

  void Layout() {
    ResetGUI();
    box = root->Box();
    CHECK(font.Load());
    Flow flow(&box, 0, &child_box);
    int tw=root->width/3, th=root->height*.05, lw=tab1.outline_w;
    flow.p.x+=1*lw-1; tab1.Layout(&flow, font, point(tw-lw*2+2, th-lw*2+2));
    flow.p.x+=2*lw-2; tab2.Layout(&flow, font, point(tw-lw*2+2, th-lw*2+2));
    flow.p.x+=2*lw-2; tab3.Layout(&flow, font, point(tw-lw*2+2, th-lw*2+2));
  }

  int Frame(LFL::Window *W, unsigned clicks, int flag) {
    W->GetInputController<BindMap>(0)->Repeat(clicks);
    W->gd->DrawMode(DrawMode::_2D);

    int mic_samples = app->audio->mic_samples;
    if (ws.AED) ws.AED->Update(mic_samples);
    if (ws.liveSG) ws.liveSG->Update(mic_samples);
    if (ws.stream) ws.stream->Update(mic_samples, app->camera->state.have_sample);

#ifdef LFL_MOBILE
    int orientation = NativeWindowOrientation();
    bool orientation_fs = orientation == 5 || orientation == 4 || orientation == 3;
    bool fullscreen = myTab == 4;
    if (orientation_fs && !fullscreen) myTab = 4;
    if (!orientation_fs && fullscreen) myTab = 1;
#endif

    GUI::Draw();
    if      (ws.myTab == 1) audio_gui->Draw();
    else if (ws.myTab == 2) video_gui->Draw();
    else if (ws.myTab == 3) {
      ScopedDrawMode sdm(W->gd, DrawMode::_3D);
      room_gui->Frame(W, clicks, flag);
    } else if (ws.myTab == 4) {
      fullscreen_gui->Draw();
      if (fullscreen_gui->close) { fullscreen_gui->close=0; SetMyTab(1); }
    }

    W->DrawDialogs();
    return 0;
  }
};

void MyWindowInitCB(Window *W) {
  W->width = 640;
  W->height = 480;
  W->caption = "fusion viewer";
}

void MyWindowStartCB(Window *W) {
  FVGUI *fv_gui = W->AddGUI(make_unique<FVGUI>(W));
  if (FLAGS_console) W->InitConsole(Callback());
  fv_gui->ws.liveSG = make_unique<LiveSpectogram>(app->asset("live"), app->asset("snap"));
  fv_gui->ws.liveSG->XForm("mel");
  fv_gui->audio_gui = W->AddGUI(make_unique<AudioGUI>(W, &fv_gui->ws));
  fv_gui->video_gui = W->AddGUI(make_unique<VideoGUI>(W, &fv_gui->ws));
  fv_gui->fullscreen_gui = W->AddGUI(make_unique<FullscreenGUI>(W, &fv_gui->ws));
  fv_gui->room_gui = make_unique<RoomGUI>(&fv_gui->ws);
  fv_gui->SetMyTab(1);
  fv_gui->ReloadAED();
  W->frame_cb = bind(&FVGUI::Frame, fv_gui, _1, _2, _3);

  BindMap *binds = W->AddInputController(make_unique<BindMap>());
  // binds->Bind(key,         callback,         arg));
  binds->Add(Key::Backquote, Bind::CB(bind(&Shell::console,  W->shell.get(), vector<string>())));
  binds->Add(Key::Quote,     Bind::CB(bind(&Shell::console,  W->shell.get(), vector<string>())));
  binds->Add(Key::Escape,    Bind::CB(bind(&Shell::quit,     W->shell.get(), vector<string>())));
  binds->Add(Key::Return,    Bind::CB(bind(&Shell::grabmode, W->shell.get(), vector<string>())));
  binds->Add(Key::LeftShift, Bind::TimeCB(bind(&Entity::RollLeft,   &fv_gui->room_gui->scene.cam, _1)));
  binds->Add(Key::Space,     Bind::TimeCB(bind(&Entity::RollRight,  &fv_gui->room_gui->scene.cam, _1)));
  binds->Add('w',            Bind::TimeCB(bind(&Entity::MoveFwd,    &fv_gui->room_gui->scene.cam, _1)));
  binds->Add('s',            Bind::TimeCB(bind(&Entity::MoveRev,    &fv_gui->room_gui->scene.cam, _1)));
  binds->Add('a',            Bind::TimeCB(bind(&Entity::MoveLeft,   &fv_gui->room_gui->scene.cam, _1)));
  binds->Add('d',            Bind::TimeCB(bind(&Entity::MoveRight,  &fv_gui->room_gui->scene.cam, _1)));
  binds->Add('q',            Bind::TimeCB(bind(&Entity::MoveDown,   &fv_gui->room_gui->scene.cam, _1)));
  binds->Add('e',            Bind::TimeCB(bind(&Entity::MoveUp,     &fv_gui->room_gui->scene.cam, _1)));

  W->shell = make_unique<Shell>(W);
  W->shell->Add("speech_client", bind(&AudioGUI::SpeechClientCmd,        fv_gui->audio_gui, _1));
  W->shell->Add("draw",          bind(&AudioGUI::DrawCmd,                fv_gui->audio_gui, _1));
  W->shell->Add("snap",          bind(&AudioGUI::SnapCmd,                fv_gui->audio_gui, _1));
  W->shell->Add("decode",        bind(&AudioGUI::DecodeCmd,              fv_gui->audio_gui, _1));
  W->shell->Add("netdecode",     bind(&AudioGUI::NetDecodeCmd,           fv_gui->audio_gui, _1));
  W->shell->Add("synth",         bind(&AudioGUI::SynthCmd,               fv_gui->audio_gui, _1));
  W->shell->Add("resynth",       bind(&AudioGUI::ResynthCmd,             fv_gui->audio_gui, _1));
  W->shell->Add("sgtf",          bind(&AudioGUI::SpectogramTransformCmd, fv_gui->audio_gui, _1));
  W->shell->Add("sgxf",          bind(&AudioGUI::SpectogramTransformCmd, fv_gui->audio_gui, _1));
  W->shell->Add("server",        bind(&AudioGUI::ServerCmd,              fv_gui->audio_gui, _1));
  W->shell->Add("startcamera",   bind(&VideoGUI::StartCameraCmd,         fv_gui->video_gui, _1));

#ifndef LFL_MOBILE
  AcousticModelFile *model = new AcousticModelFile();
  if (model->Open("AcousticModel", Asset::FileName("").c_str()) < 0) return ERROR(-1, "am read ", Asset::FileName(""));
  if (!(fv_gui->audio_gui->decodeModel = AcousticModel::FromModel1(model, true))) return ERROR(-1, "model create failed");
  AcousticModel::ToCUDA(model);

  PronunciationDict::Instance();
  VoiceModel *voice = (fv_gui->audio_gui->voice = make_unique<VoiceModel>()).get();
  if (voice->Read(Asset::FileName("").c_str()) < 0) return ERROR(-1, "voice read ", Asset::FileName(""));

  fv_gui->ws.stream = new HTTPServer::StreamResource("flv", 32000, 300000);
  my_app->httpd->AddURL("/stream.flv", fv_gui->ws.stream);
#endif
}

}; // namespace LFL
using namespace LFL;

extern "C" void MyAppCreate(int argc, const char* const* argv) {
  FLAGS_target_fps = 30;
  FLAGS_enable_video = FLAGS_enable_audio = FLAGS_enable_input = FLAGS_enable_network = FLAGS_enable_camera = FLAGS_console = true;
  FLAGS_console_font = "Nobile.ttf";
  FLAGS_font_flag = FLAGS_console_font_flag = 0;
  FLAGS_chans_in = -1;
  app = new Application(argc, argv);
  app->focused = Window::Create();
  my_app = new MyAppState();
  app->window_start_cb = MyWindowStartCB;
  app->window_init_cb = MyWindowInitCB;
  app->window_init_cb(app->focused);
  app->exit_cb = []{ delete my_app; };
}

extern "C" int MyAppMain() {
  if (app->Create(__FILE__)) return -1;
  if (FLAGS_font_engine == "atlas") FLAGS_font = "Nobile.ttf";

  if (app->Init()) return -1;
  app->focused->gd->default_draw_mode = DrawMode::_3D;

  // app->asset.Add(name, texture, scale, translate, rotate, geometry, 0, 0, 0);
  app->asset.Add("axis", "", 0, 0, 0, nullptr, nullptr, 0, 0, bind(&glAxis, _1, _2, _3));
  app->asset.Add("grid", "", 0, 0, 0, Grid::Grid3D().release(), nullptr, 0, 0);
  app->asset.Add("room", "", 0, 0, 0, nullptr, nullptr, 0, 0, bind(&glRoom, _1, _2, _3));
  app->asset.Add("arrow", "", .005, 1, -90, "arrow.obj", nullptr, 0);
  app->asset.Add("snap", "", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("live", "", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("browser", "", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("camera", "", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("camfx", "", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("sbg", "spectbg.png", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("sgloss", "spectgloss.png", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("onoff0", "onoff0.png", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("onoff0hover", "onoff0hover.png", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("onoff1", "onoff1.png", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("onoff1hover", "onoff1hover.png", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("but0", "but0.png", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("but1", "but1.png", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("butplay", "play-icon.png", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("butclose", "close-icon.png", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Add("butinfo", "info-icon.png", 0, 0, 0, nullptr, nullptr, 0, 0);
  app->asset.Load();

  // app->soundasset.Add(name, filename, ringbuf, channels, sample_rate, seconds);
  app->soundasset.Add("draw", "Draw.wav", nullptr, 0, 0, 0);
  app->soundasset.Add("snap", "", new RingSampler(FLAGS_sample_rate*FLAGS_sample_secs), 1, FLAGS_sample_rate, FLAGS_sample_secs);
  app->soundasset.Load();

#ifndef LFL_MOBILE
  HTTPServer *httpd = my_app->httpd = new HTTPServer(4040, false);
  if (app->net->Enable(httpd)) return -1;
  httpd->AddURL("/favicon.ico", new HTTPServer::FileResource(Asset::FileName("icon.ico"), "image/x-icon"));
  httpd->AddURL("/test.flv", new HTTPServer::FileResource(Asset::FileName("test.flv"), "video/x-flv"));
  httpd->AddURL("/mediaplayer.swf", new HTTPServer::FileResource(Asset::FileName("mediaplayer.swf"), "application/x-shockwave-flash"));
  httpd->AddURL("/", new HTTPServer::StringResource
                ("text/html; charset=UTF-8",
                 "<html><h1>Web Page</h1>\r\n"
                 "<a href=\"http://www.google.com\">google</a><br/>\r\n"
                 "<h2>stream</h2>\r\n"
                 "<embed src=\"/mediaplayer.swf\" width=\"320\" height=\"240\" \r\n"
                 "       allowscriptaccess=\"always\" allowfullscreen=\"true\" \r\n"
                 "       flashvars=\"width=320&height=240&file=/stream.flv\" />\r\n"
                 "</html>\r\n"));
#endif

  app->StartNewWindow(app->focused);
  return app->Main();
}
