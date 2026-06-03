import unittest
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import task as tm
from app.models.schema import MaterialInfo, VideoParams

resources_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")

class TestTaskService(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_task_local_materials(self):
        task_id = "00000000-0000-0000-0000-000000000000"
        video_materials=[]
        for i in range(1, 4):
            video_materials.append(MaterialInfo(
                provider="local",
                url=os.path.join(resources_dir, f"{i}.png"),
                duration=0
            ))

        params = VideoParams(
            video_subject="金钱的作用",
            video_script="金钱不仅是交换媒介，更是社会资源的分配工具。它能满足基本生存需求，如食物和住房，也能提供教育、医疗等提升生活品质的机会。拥有足够的金钱意味着更多选择权，比如职业自由或创业可能。但金钱的作用也有边界，它无法直接购买幸福、健康或真诚的人际关系。过度追逐财富可能导致价值观扭曲，忽视精神层面的需求。理想的状态是理性看待金钱，将其作为实现目标的工具而非终极目的。",
            video_terms="money importance, wealth and society, financial freedom, money and happiness, role of money",
            video_aspect="9:16",
            video_concat_mode="random",
            video_transition_mode="None",
            video_clip_duration=3,
            video_count=1,
            video_source="local",
            video_materials=video_materials,
            video_language="",
            voice_name="zh-CN-XiaoxiaoNeural-Female",
            voice_volume=1.0,
            voice_rate=1.0,
            bgm_type="random",
            bgm_file="",
            bgm_volume=0.2,
            subtitle_enabled=True,
            subtitle_position="bottom",
            custom_position=70.0,
            font_name="MicrosoftYaHeiBold.ttc",
            text_fore_color="#FFFFFF",
            text_background_color=True,
            font_size=60,
            stroke_color="#000000",
            stroke_width=1.5,
            n_threads=2,
            paragraph_number=1
        )
        result = tm.start(task_id=task_id, params=params)
        print(result)

    def test_generate_final_videos_uses_default_output_filename(self):
        params = SimpleNamespace(
            video_count=1,
            video_concat_mode="random",
            video_transition_mode=None,
            video_clip_duration=3,
            n_threads=2,
            video_aspect="9:16",
        )

        with patch.object(tm.utils, "task_dir", return_value="/tmp/task-dir"), patch.object(
            tm.video, "combine_videos"
        ), patch.object(tm.video, "generate_video"), patch.object(
            tm.sm.state, "update_task"
        ):
            final_video_paths, combined_video_paths = tm.generate_final_videos(
                task_id="task-id",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="subtitle.srt",
                video_title="这才是真正值得关注的变化",
            )
        self.assertEqual(final_video_paths, ["/tmp/task-dir/final-1.mp4"])
        self.assertEqual(combined_video_paths, ["/tmp/task-dir/combined-1.mp4"])

    def test_get_video_materials_loads_storage_local_videos_en_when_params_empty(self):
        params = SimpleNamespace(
            video_source="local",
            video_materials=[],
            video_clip_duration=3,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            local_videos_dir = os.path.join(temp_dir, "local_videos_en")
            os.makedirs(local_videos_dir)
            local_video_path = os.path.join(local_videos_dir, "sample.mp4")
            with open(local_video_path, "wb") as fp:
                fp.write(b"fake")

            def _fake_preprocess(materials, clip_duration):
                self.assertEqual(clip_duration, 3)
                self.assertEqual(len(materials), 1)
                self.assertEqual(materials[0].url, local_video_path)
                return materials

            with patch.object(tm.utils, "storage_dir", return_value=local_videos_dir), patch.object(
                tm.video, "preprocess_video", side_effect=_fake_preprocess
            ), patch.object(tm.sm.state, "update_task"):
                result = tm.get_video_materials(
                    task_id="task-id",
                    params=params,
                    video_terms=[],
                    audio_duration=60,
                )

        self.assertEqual(result, [local_video_path])
    

if __name__ == "__main__":
    unittest.main() 