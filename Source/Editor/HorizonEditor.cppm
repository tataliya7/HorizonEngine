export module HorizonEditor;

export namespace HE
{
	class HorizonEditor
	{
	public:

		static HorizonEditor* Instance;
		static HorizonEditor* GetInstance()
		{
			return Instance;
		}

		HorizonEditor();
		virtual ~HorizonEditor();

		HorizonEditor(HorizonEditor&) = delete;
		HorizonEditor(const HorizonEditor&) = delete;
		HorizonEditor& operator=(HorizonEditor&) = delete;
		HorizonEditor& operator=(const HorizonEditor&) = delete;

		void Tick()
		{
			UpdateFPS();
			switch (m_SceneState)
			{
			case SceneViewportState::Edit:
			{
				m_EditorCamera.SetActive(m_AllowViewportCameraEvents);
				m_EditorCamera.OnUpdate(ts);
				m_EditorScene->OnRenderEditor(m_ViewportRenderer, ts, m_EditorCamera);
				
				OnRender2D();

				if (const auto& project = Project::GetActive(); project && project->GetConfig().EnableAutoSave)
				{
					m_TimeSinceLastSave += ts;
					if (m_TimeSinceLastSave > project->GetConfig().AutoSaveIntervalSeconds)
					{
						SaveSceneAuto();
					}
				}
				break;
			}
			case SceneViewportState::Play:
			{
				m_RuntimeScene->OnUpdate(ts);
				m_RuntimeScene->OnRenderRuntime(m_ViewportRenderer, ts);

				for (auto& fn : m_PostSceneUpdateQueue)
					fn();
				m_PostSceneUpdateQueue.clear();
				break;
			}
			case SceneViewportState::Pause:
			{
				editorViewportCamera;
				m_EditorCamera.OnUpdate(ts);
				m_RuntimeScene->OnRenderRuntime(m_ViewportRenderer, ts);
				break;
			}
			}
		}

	private:

	};

	int HorizonEditorMain()
	{
		HE::LogSystemInit();
		HE::JobSystemInit(HE::GetNumberOfProcessors(), HE_JOB_SYSTEM_NUM_FIBIERS, HE_JOB_SYSTEM_FIBER_STACK_SIZE);
		int exitCode = EXIT_SUCCESS;
		HorizonEditor* editor = new HorizonEditor();
		bool result = editor->Init();
		if (result)
		{
			exitCode = editor->Run();
		}
		else
		{
			exitCode = EXIT_FAILURE;
		}
		editor->Exit();
		HE::JobSystemExit();
		HE::LogSystemExit();
		delete editor;
		return exitCode;
	}
}