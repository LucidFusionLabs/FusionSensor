# $Id: CMakeLists.txt 1320 2014-10-26 08:12:52Z justin $

lfl_project(fv)

lfl_add_target(fv fv.cpp ${LFL_RECOGNITION_SOURCE})
target_compile_definitions(fv PUBLIC ${LFL_APP_DEF} -DLFL_OPENCV)
target_include_directories(fv PUBLIC ${LFL_APP_INCLUDE} ${OPENCV_INCLUDE})
target_link_libraries(fv ${LFL_APP_LIB} ${LFL_APP_PLATFORM} ${LFL_APP_GRAPHICS}
                      ${LFL_APP_AUDIO} ${LFL_APP_CAMERA} app_opencv_matrix app_ffmpeg_fft
                      app_ffmpeg_resampler app_ffmpeg_loader app_iconv_convert
                      app_libpng_png app_null_jpeg app_null_gif app_null_css app_null_ttf
                      app_null_ssl app_ffmpeg_stream)
lfl_post_build_start(fv fv FusionViewer)

add_executable(fs fs.cpp ${LFL_RECOGNITION_SOURCE})
target_include_directories(fs PUBLIC ${LFL_APP_INCLUDE})
target_link_libraries(fs ${LFL_APP_LIB} app_null_platform app_null_graphics
                      app_null_audio app_null_camera app_opencv_matrix app_ffmpeg_fft
                      app_simple_resampler app_simple_loader app_iconv_convert
                      app_null_png app_null_jpeg app_null_gif app_null_css app_null_ttf
                      app_null_ssl)
lfl_post_build_start(fs fs FusionServer)

