# Google Cloud Console OAuth 2.0 설정 가이드

## 1단계: Google Cloud Console 접속
1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. Google 계정으로 로그인

## 2단계: 프로젝트 생성 또는 선택
1. 상단의 프로젝트 선택 드롭다운 클릭
2. **"새 프로젝트"** 클릭
3. 프로젝트 이름 입력 (예: "Book Organizer OAuth")
4. **"만들기"** 클릭
5. 프로젝트가 생성되면 선택

## 3단계: Google+ API 활성화
1. 왼쪽 메뉴에서 **"API 및 서비스"** > **"라이브러리"** 클릭
2. 검색창에 **"Google+ API"** 입력
3. **"Google+ API"** 클릭
4. **"사용"** 버튼 클릭

## 4단계: OAuth 동의 화면 설정
1. 왼쪽 메뉴에서 **"API 및 서비스"** > **"OAuth 동의 화면"** 클릭
2. **"외부"** 선택 (개인 Google 계정 사용)
3. **"만들기"** 클릭
4. 필수 정보 입력:
   - **앱 이름**: "Book Organizer"
   - **사용자 지원 이메일**: 본인 이메일
   - **개발자 연락처 정보**: 본인 이메일
5. **"저장 후 계속"** 클릭
6. **"범위"** 페이지에서 **"저장 후 계속"** 클릭
7. **"테스트 사용자"** 페이지에서 **"저장 후 계속"** 클릭
8. **"요약"** 페이지에서 **"대시보드로 돌아가기"** 클릭

## 5단계: OAuth 2.0 Client ID 생성
1. 왼쪽 메뉴에서 **"API 및 서비스"** > **"사용자 인증 정보"** 클릭
2. **"+ 사용자 인증 정보 만들기"** 클릭
3. **"OAuth 클라이언트 ID"** 선택
4. **애플리케이션 유형**: **"웹 애플리케이션"** 선택
5. **이름**: "Book Organizer Web App" 입력
6. **승인된 리디렉션 URI** 섹션에서 **"+ URI 추가"** 클릭
7. 다음 URI 추가:
   ```
   https://qqkkygzogkerdixzratb.supabase.co/auth/v1/callback
   ```
8. **"만들기"** 클릭

## 6단계: Client ID와 Secret 복사
생성 완료 후 팝업에서 다음 정보를 복사:
- **클라이언트 ID**: `123456789-abcdefg.apps.googleusercontent.com` 형태
- **클라이언트 보안 비밀**: `ABCDEFGHIJKLMNOP` 형태

## 7단계: Supabase에 설정 적용
1. [Supabase Dashboard](https://supabase.com/dashboard) 접속
2. 프로젝트 선택
3. **"Authentication"** > **"Providers"** 클릭
4. **"Google"** 찾아서 토글 활성화
5. 다음 정보 입력:
   - **Client ID**: 6단계에서 복사한 클라이언트 ID
   - **Client Secret**: 6단계에서 복사한 클라이언트 시크릿
6. **"Save"** 클릭

## 8단계: 테스트
1. Book Organizer 앱 실행
2. **"Sign in with Google"** 버튼 클릭
3. Google 로그인 완료
4. 앱으로 리디렉션 확인

## 문제 해결
- **"redirect_uri_mismatch"** 오류: Supabase 리디렉션 URI가 정확한지 확인
- **"invalid_client"** 오류: Client ID와 Secret이 정확한지 확인
- **"access_denied"** 오류: OAuth 동의 화면 설정 확인

## 보안 주의사항
- Client Secret은 절대 공개하지 마세요
- 프로덕션에서는 HTTPS 사용 필수
- 정기적으로 자격 증명 갱신 권장
