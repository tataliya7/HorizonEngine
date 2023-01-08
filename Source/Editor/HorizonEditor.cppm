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

		HorizonEditor(const HorizonEditor&) = delete;
		HorizonEditor& operator=(const HorizonEditor&) = delete;

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