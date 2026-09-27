# 增量包应用说明（Goal1 / Goal2 共用 `tasks/real_pipeline.py`）

`tasks/real_pipeline.py` 是**插件接线表**：它列出任务链上每个位置用哪个真实插件、
数据集级任务用哪个。**每交付一个新 Goal，这个文件都会更新一次**，并且它同时包含
此前已交付的所有插件接线（Goal2 包里这一版 = Goal1 + Goal2）。

因此增量包存在“谁最后解压谁生效”的覆盖关系：

```text
正确顺序：goal1_upload.zip  →  goal2_upload.zip     # Goal2 一定要最后覆盖
错误顺序：goal2_upload.zip  →  goal1_upload.zip     # real_pipeline.py 被退回旧版
```

## 症状

解压顺序错误时，代码文件都在，但管线绑定的是 Dummy：

```text
FAIL: test_real_pipeline_binds_goal2_tasks
AssertionError: <tasks.dummy.study_tasks.DummyStitchedTask object ...> is not an instance of
                <class 'tasks.goal2_stitched.task.Goal2StitchedTask'>
```

后果：`IsStitchedProb` / `duplicate_pairs.jsonl` 全是占位值（静默丢分），但服务能正常起。

## 修法（三选一）

```bash
cd <仓库根目录>          # 例：/2026aicompetition/workspace/dcs/Glioma_recognition

# A) 只把接线表恢复成 Goal2 那一版（最快）
unzip -o /path/to/goal2_upload.zip tasks/real_pipeline.py -d .

# B) 重新按正确顺序解压两个增量包
unzip -o /path/to/goal1_upload.zip -d .
unzip -o /path/to/goal2_upload.zip -d .

# C) 直接用合并包（含 Goal1+Goal2 全部文件，不存在顺序问题）
unzip -o /path/to/goal12_upload.zip -d .
```

## 校验

```bash
# 1) 接线表里应当是真实插件，而不是 Dummy
grep -nE "Goal2StitchedTask|Goal2DuplicateRecorder|DummyStitchedTask|DummyDuplicateTask" tasks/real_pipeline.py

# 2) 契约测试 + 配置自检（会显式检查绑定，并把修法打在报错里）
python -m unittest tests.contracts.test_goal2_contract -v
bash tasks/goal2_stitched/verify.sh --skip-service
bash tasks/goal2_duplicate/verify.sh --skip-service
```

期望输出（阶段 2）里能看到：

```text
任务链： [('goal1','Goal1AuthenticityTask'), ('goal2_stitched','Goal2StitchedTask'), ...]
数据集级任务： Goal2DuplicateRecorder
```
