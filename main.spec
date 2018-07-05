# -*- mode: python -*-

block_cipher = None

added_files = [('/Users/jon/PycharmProjects/zbow/venv/lib/python3.6/site-packages/vispy', 'vispy'),
               ('/Users/jon/PyCharmProjects/zbow/bin/logo.png', 'bin'),
	       ('/Users/jon/PyCharmProjects/zbow/bin/logicle' , 'bin/logicle')]

added_imports = ['scipy._lib.messagestream','pandas._libs.tslibs.timedeltas', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree', 'sklearn.tree._utils', 'vispy.glsl', 'PyQt5']

a = Analysis(['main.py'],
             pathex=['/Users/jon/PycharmProjects/zbow/venv/lib/python3.6/site-packages', '/Users/jon/PycharmProjects/zbow'],
             binaries=[],
             datas=added_files,
             hiddenimports=added_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='zbow_analysis_app',
          debug=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='zbow_analysis_app')
